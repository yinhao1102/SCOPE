import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math
from typing import Optional


# Initial RMS scale of the adapted magnitude relative to the frozen weight.
GATE_UPDATE_RMS_RATIO = 0.01
CALIBRATED_GATE_MODES = ('ab_norm_calibrated', 'ab_norm_learned_calibrated',
                         'ab_norm_rank_calibrated', 'ab_norm_strength_calibrated')
LEARNED_GATE_MODE = 'ab_norm_learned_calibrated'
RANK_LEARNED_GATE_MODES = (LEARNED_GATE_MODE, 'ab_norm_rank_calibrated')
STRENGTH_LEARNED_GATE_MODES = (LEARNED_GATE_MODE, 'ab_norm_strength_calibrated')

# This family changes only the existing magnitude gate. The four factors,
# sign direction, adapter placement and training objective remain unchanged.
CHANNEL_GATE_MODES = (
    'ab_norm_out_scalar', 'ab_norm_out_channel', 'ab_norm_out_channel_budget',
    'ab_norm_out_channel_shared', 'ab_norm_out_channel_bounded',
    'ab_norm_in_channel_budget',
)
STRUCTURED_GATE_MODES = (*CHANNEL_GATE_MODES, 'ab_norm_scalar_bounded')
CHANNEL_LEARNED_GATE_MODES = tuple(m for m in CHANNEL_GATE_MODES if m != 'ab_norm_out_scalar')
CHANNEL_SHARED_GATE_MODES = ('ab_norm_out_channel_shared',)
CHANNEL_BUDGET_GATE_MODES = (
    'ab_norm_out_channel_budget', 'ab_norm_out_channel_shared',
    'ab_norm_out_channel_bounded', 'ab_norm_in_channel_budget',
)
BOUNDED_STRENGTH_GATE_MODES = ('ab_norm_scalar_bounded', 'ab_norm_out_channel_bounded')
STRENGTH_LEARNED_GATE_MODES += STRUCTURED_GATE_MODES


def clipped_ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Hard sign in forward, clipped straight-through gradient in backward."""
    clipped = x.clamp(-1.0, 1.0)
    return x.sign().detach() + clipped - clipped.detach()


def rank_gate(A: torch.Tensor, B: torch.Tensor, mode='rms', eps=1e-6,
              logits: Optional[torch.Tensor] = None):
    """Return one gate per rank; A is [rank, in], B is [out, rank].

    ``none`` removes only the gate from the dual-branch update. ``rms`` is
    the original SCOPE statistic. ``ab_norm`` is an optional diagnostic:
    its unnormalized score is invariant to positive reciprocal rescaling
    of each corresponding A row and B column. It is not a proven retrieval
    contribution measure.
    """
    if mode == 'none':
        return A.new_ones(A.shape[0])
    if mode == 'rms':
        score = torch.sqrt(torch.mean(A ** 2, dim=1) + eps)
    elif mode == 'ab_norm' or mode in CALIBRATED_GATE_MODES or mode in STRUCTURED_GATE_MODES:
        # vector_norm defines a finite zero-vector subgradient; unlike a
        # bare sqrt(sum(x**2)), it does not introduce NaNs when B is zero.
        score = torch.linalg.vector_norm(A, dim=1) * torch.linalg.vector_norm(B, dim=0)
        if mode in RANK_LEARNED_GATE_MODES:
            if logits is None or logits.shape != score.shape:
                raise ValueError('Learned rank gate requires one logit per rank')
            # Zero logits reproduce AB norm. Bounded relative weights avoid
            # an unbounded gate competing with the low-rank factor scale.
            score = score * logits.tanh().exp()
    else:
        raise ValueError(f"Unsupported gate mode: {mode}")
    return score / (score.mean() + eps)


def calibrate_gate_magnitude(magnitude: torch.Tensor, reference: torch.Tensor,
                             scaling: float, ratio=GATE_UPDATE_RMS_RATIO):
    """Calibrate only the magnitude gate; keep the original STE direction.

    The scaled gate has RMS ``ratio * RMS(reference)`` for nonzero magnitude.
    Multiplication by sign can only reduce this norm, and D=0 still gives an
    exactly zero update. Norm accumulation is outside autocast precision.
    There is no additive direction path or change to the pretrained weight.
    """
    if scaling <= 0 or (isinstance(ratio, (float, int)) and ratio <= 0):
        raise ValueError('Gate calibration requires positive scaling and ratio')
    dtype = torch.float64 if magnitude.dtype == torch.float64 else torch.float32
    value = magnitude.to(dtype)
    rms = torch.linalg.vector_norm(value) / math.sqrt(value.numel())
    base = reference.detach().to(dtype)
    target = ratio * torch.linalg.vector_norm(base) / math.sqrt(base.numel())
    calibrated = value * (target / (scaling * rms.clamp_min(1e-8)))
    return calibrated.to(magnitude.dtype)


def learned_gate_ratio(strength, mode=None):
    """Positive layer/modality gate strength, initialized to the 1% budget.

    A learned strength can grow beyond the calibration-only ablation's fixed
    budget. Softplus is parameterized in ratio units so its initial gradient
    is not multiplied by the small initial ratio.
    """
    if mode in BOUNDED_STRENGTH_GATE_MODES:
        # Preserve both the 1% initial value and the old softplus derivative.
        # The cap is 4% of the frozen-weight RMS, not a change to the loss.
        center = math.log(math.expm1(1.))
        slope = (1. - math.exp(-1.)) / .75
        return 4. * GATE_UPDATE_RMS_RATIO * torch.sigmoid(
            slope * (strength / GATE_UPDATE_RMS_RATIO - center) - math.log(3.))
    return GATE_UPDATE_RMS_RATIO * F.softplus(strength / GATE_UPDATE_RMS_RATIO)


def channel_gate_axis(mode, output_axis):
    return 1 - output_axis if mode == 'ab_norm_in_channel_budget' else output_axis


def calibrate_channel_gate(magnitude, reference_aligned, scaling, ratio,
                           mode, output_axis, channel_logits=None):
    """Allocate the update budget along existing weight channels.

    ``reference_aligned`` uses the same matrix orientation as ``magnitude``.
    Frozen channel norms provide the scale prior. Learned channel weights
    redistribute that budget; budget variants preserve its overall RMS.
    """
    if mode not in CHANNEL_GATE_MODES or output_axis not in (0, 1) or scaling <= 0:
        raise ValueError('Invalid channel gate configuration')
    if reference_aligned.shape != magnitude.shape:
        raise ValueError('Channel gate requires aligned matrix orientations')
    axis = channel_gate_axis(mode, output_axis)
    reduce_axis = 1 - axis
    dtype = torch.float64 if magnitude.dtype == torch.float64 else torch.float32
    value = magnitude.to(dtype)
    base = reference_aligned.detach().to(dtype)
    width = math.sqrt(value.shape[reduce_axis])
    value_rms = torch.linalg.vector_norm(value, dim=reduce_axis, keepdim=True) / width
    reference_rms = torch.linalg.vector_norm(base, dim=reduce_axis, keepdim=True) / width
    if mode in CHANNEL_LEARNED_GATE_MODES:
        if channel_logits is None or channel_logits.shape != (value.shape[axis],):
            raise ValueError('One learned channel gate is required per selected channel')
        shape = [1, 1]
        shape[axis] = value.shape[axis]
        relative = (channel_logits.to(dtype) / GATE_UPDATE_RMS_RATIO).tanh().exp().reshape(shape)
        if mode in CHANNEL_BUDGET_GATE_MODES:
            # Weighted normalization keeps RMS(delta) controlled by the same
            # global strength as before; channel gates select its allocation.
            weighted = reference_rms * relative
            relative = relative * (torch.linalg.vector_norm(reference_rms)
                                   / torch.linalg.vector_norm(weighted).clamp_min(1e-12))
    else:
        relative = 1.
    target = ratio * reference_rms * relative
    result = value * (target / (scaling * value_rms.clamp_min(1e-8)))
    return result.to(magnitude.dtype)


class GDLoRAParameter(nn.Module):
    def __init__(self, orig_param, A_param, B_param, C_param, D_param,
                 lora_alpha: float, r1, r2, gate_eps=1e-6, gate_mode='ab_norm_out_channel',
                 direction_residual: float = 0.0):
        super().__init__()
        self.orig_param = orig_param
        self.lora_A_param = A_param
        self.lora_B_param = B_param
        self.lora_C_param = C_param
        self.lora_D_param = D_param
        self.lora_alpha = lora_alpha
        self.r1 = r1
        self.r2 = r2
        self.scaling = self.lora_alpha / math.sqrt(self.r1 * self.r2)

        self.gate_eps = gate_eps
        self.gate_mode = gate_mode
        self.direction_residual = direction_residual
        if gate_mode in RANK_LEARNED_GATE_MODES:
            self.lora_gate_logits = nn.Parameter(A_param.new_zeros(r1))
        else:
            self.register_parameter('lora_gate_logits', None)
        if gate_mode in STRENGTH_LEARNED_GATE_MODES:
            self.lora_gate_strength = nn.Parameter(A_param.new_tensor(
                GATE_UPDATE_RMS_RATIO * math.log(math.expm1(1.))))
        else:
            self.register_parameter('lora_gate_strength', None)
        if gate_mode in CHANNEL_LEARNED_GATE_MODES:
            axis = channel_gate_axis(gate_mode, 0)
            self.lora_gate_channel = nn.Parameter(A_param.new_zeros(orig_param.shape[axis]))
        else:
            self.register_parameter('lora_gate_channel', None)

    def get_gate_A_rms(self):
        """
        根据所选模式生成 rank gate；学习模式另含可学习 logits。
        A: [r1, in_features]
        返回:
            gate: [r1]
        """
        # Retain this legacy accessor name for old analysis scripts.
        return rank_gate(self.lora_A_param, self.lora_B_param,
                         mode=self.gate_mode, eps=self.gate_eps,
                         logits=self.lora_gate_logits)

    def combined_matrix(self):
        """
        组合原始参数矩阵和 LoRA 矩阵，得到一个合并后的矩阵
        """
        gate_A = self.get_gate_A_rms()                      # [r1]
        A = self.lora_A_param * gate_A.unsqueeze(1)        # [r1, in_features]

        magnitude = F.relu(A.T @ self.lora_B_param.T)
        if self.gate_mode in CHANNEL_GATE_MODES:
            magnitude = calibrate_channel_gate(magnitude, self.orig_param, self.scaling,
                learned_gate_ratio(self.lora_gate_strength, self.gate_mode),
                self.gate_mode, output_axis=0, channel_logits=self.lora_gate_channel)
        elif self.gate_mode in CALIBRATED_GATE_MODES or self.gate_mode == 'ab_norm_scalar_bounded':
            ratio = (learned_gate_ratio(self.lora_gate_strength)
                     if self.gate_mode in STRENGTH_LEARNED_GATE_MODES else GATE_UPDATE_RMS_RATIO)
            if self.gate_mode == 'ab_norm_scalar_bounded':
                ratio = learned_gate_ratio(self.lora_gate_strength, self.gate_mode)
            magnitude = calibrate_gate_magnitude(magnitude, self.orig_param, self.scaling, ratio)
        direction_matrix = self.lora_C_param.T @ self.lora_D_param.T
        dual_update = magnitude * clipped_ste_sign(direction_matrix)
        if self.direction_residual != 0.0:
            dual_update = dual_update + self.direction_residual * direction_matrix
        lora_contribution = dual_update * self.scaling

        return self.orig_param + lora_contribution


class GDLoRAMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        lora_r1=16,
        lora_r2=16,
        num_loras=16,
        lora_alpha: float = 1.0,
        device=None,
        dtype=None,
        gate_eps=1e-6,
        gate_mode='ab_norm_out_channel',
        direction_residual: float = 0.0,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lora_alpha = lora_alpha
        self.r1 = lora_r1
        self.r2 = lora_r2
        self.gate_eps = gate_eps
        self.gate_mode = gate_mode
        self.direction_residual = direction_residual

        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"

        # 原始注意力参数
        self.in_proj_weight = nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.in_proj_bias = nn.Parameter(
            torch.empty(3 * embed_dim, **factory_kwargs)
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        # ========== 关键修改：必须用 ParameterList 注册 ==========
        # in_proj 对应的 LoRA
        self.loras_A_in = nn.ParameterList([
            nn.Parameter(torch.zeros((self.r1, 3 * embed_dim), **factory_kwargs))
            for _ in range(num_loras)
        ])
        self.loras_B_in = nn.ParameterList([
            nn.Parameter(torch.zeros((embed_dim, self.r1), **factory_kwargs))
            for _ in range(num_loras)
        ])
        self.loras_C_in = nn.ParameterList([
            nn.Parameter(torch.zeros((self.r2, 3 * embed_dim), **factory_kwargs))
            for _ in range(num_loras)
        ])
        self.loras_D_in = nn.ParameterList([
            nn.Parameter(torch.zeros((embed_dim, self.r2), **factory_kwargs))
            for _ in range(num_loras)
        ])

        # out_proj 对应的 LoRA
        self.loras_A_out = nn.ParameterList([
            nn.Parameter(torch.zeros((self.r1, embed_dim), **factory_kwargs))
            for _ in range(num_loras)
        ])
        self.loras_B_out = nn.ParameterList([
            nn.Parameter(torch.zeros((embed_dim, self.r1), **factory_kwargs))
            for _ in range(num_loras)
        ])
        self.loras_C_out = nn.ParameterList([
            nn.Parameter(torch.zeros((self.r2, embed_dim), **factory_kwargs))
            for _ in range(num_loras)
        ])
        self.loras_D_out = nn.ParameterList([
            nn.Parameter(torch.zeros((embed_dim, self.r2), **factory_kwargs))
            for _ in range(num_loras)
        ])

        # Wrap each modality-specific update as a GD-LoRA parameterization.
        self.in_proj_weight_withloras = nn.ModuleList([
            GDLoRAParameter(
                self.in_proj_weight,
                self.loras_A_in[i],
                self.loras_B_in[i],
                self.loras_C_in[i],
                self.loras_D_in[i],
                self.lora_alpha,
                self.r1,
                self.r2,
                gate_eps=self.gate_eps,
                gate_mode=self.gate_mode,
                direction_residual=self.direction_residual,
            ) for i in range(num_loras)
        ])
        self.out_proj_weight_withloras = nn.ModuleList([
            GDLoRAParameter(
                self.out_proj.weight,
                self.loras_A_out[i],
                self.loras_B_out[i],
                self.loras_C_out[i],
                self.loras_D_out[i],
                self.lora_alpha,
                self.r1,
                self.r2,
                gate_eps=self.gate_eps,
                gate_mode=self.gate_mode,
                direction_residual=self.direction_residual,
            ) for i in range(num_loras)
        ])
        if gate_mode in CHANNEL_SHARED_GATE_MODES:
            for adapters in (self.in_proj_weight_withloras, self.out_proj_weight_withloras):
                shared_channel = adapters[0].lora_gate_channel
                for adapter in adapters[1:]:
                    adapter.lora_gate_channel = shared_channel

        # ========== 正确冻结基础权重 ==========
        self.in_proj_weight.requires_grad = False
        self.in_proj_bias.requires_grad = False
        self.out_proj.weight.requires_grad = False
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad = False

        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False
        self.dropout = 0.0

        self.reset_parameters()

        # 可选：LoRA 参数保持 float32，增强稳定性
        for name, param in self.named_parameters():
            if ("loras_" in name) or ("gate" in name):
                param.data = param.data.float()

    def reset_parameters(self):
        # 原始 MHA 参数初始化
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        # LoRA 初始化
        for i in range(len(self.loras_A_in)):
            # D keeps the initial delta at zero; a small B lets D receive
            # a gradient through the straight-through sign on the first step.
            nn.init.kaiming_uniform_(self.loras_A_in[i], a=math.sqrt(5))
            nn.init.normal_(self.loras_B_in[i], std=1e-4)
            nn.init.kaiming_uniform_(self.loras_C_in[i], a=math.sqrt(5))
            nn.init.zeros_(self.loras_D_in[i])

            nn.init.kaiming_uniform_(self.loras_A_out[i], a=math.sqrt(5))
            nn.init.normal_(self.loras_B_out[i], std=1e-4)
            nn.init.kaiming_uniform_(self.loras_C_out[i], a=math.sqrt(5))
            nn.init.zeros_(self.loras_D_out[i])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        lora_index: int = 0
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        in_proj_weight = self.in_proj_weight_withloras[lora_index].combined_matrix()
        out_proj_weight = self.out_proj_weight_withloras[lora_index].combined_matrix()

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            out_proj_weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        return attn_output, attn_output_weights


class GDLoRALayer:
    def __init__(
        self,
        r1: int,
        r2: int,
        lora_alpha: float,
        lora_dropout: float,
        direction_residual: float = 0.0,
    ):
        self.r1 = r1
        self.r2 = r2
        self.lora_alpha = lora_alpha
        self.direction_residual = direction_residual
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class GDLoRALinear(nn.Linear, GDLoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r1: int = 16,
        lora_r2: int = 16,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        num_loras: int = 16,
        gate_eps: float = 1e-6,
        gate_mode: str = 'ab_norm_out_channel',
        direction_residual: float = 0.0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GDLoRALayer.__init__(
            self,
            r1=lora_r1,
            r2=lora_r2,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            direction_residual=direction_residual,
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.gate_eps = gate_eps
        self.gate_mode = gate_mode

        if lora_r1 > 0 and lora_r2 > 0:
            self.loras_A = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((lora_r1, in_features)))
                for _ in range(num_loras)
            ])
            self.loras_B = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((out_features, lora_r1)))
                for _ in range(num_loras)
            ])
            self.loras_C = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((lora_r2, in_features)))
                for _ in range(num_loras)
            ])
            self.loras_D = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((out_features, lora_r2)))
                for _ in range(num_loras)
            ])
            if gate_mode in RANK_LEARNED_GATE_MODES:
                self.loras_gate_logits = nn.ParameterList([
                    nn.Parameter(self.weight.new_zeros(lora_r1))
                    for _ in range(num_loras)
                ])
            if gate_mode in STRENGTH_LEARNED_GATE_MODES:
                self.loras_gate_strengths = nn.ParameterList([
                    nn.Parameter(self.weight.new_tensor(
                        GATE_UPDATE_RMS_RATIO * math.log(math.expm1(1.))))
                    for _ in range(num_loras)
                ])
            if gate_mode in CHANNEL_LEARNED_GATE_MODES:
                channels = in_features if gate_mode == 'ab_norm_in_channel_budget' else out_features
                self.loras_gate_channels = nn.ParameterList([
                    nn.Parameter(self.weight.new_zeros(channels))
                    for _ in range(num_loras)
                ])
                if gate_mode in CHANNEL_SHARED_GATE_MODES:
                    for index in range(1, num_loras):
                        self.loras_gate_channels[index] = self.loras_gate_channels[0]

            self.scaling = self.lora_alpha / math.sqrt(self.r1 * self.r2)

            # 冻结基础权重
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'loras_A'):
            for lora_A in self.loras_A:
                nn.init.normal_(lora_A, std=0.02)
            for lora_B in self.loras_B:
                nn.init.normal_(lora_B, std=1e-4)
            for lora_C in self.loras_C:
                nn.init.normal_(lora_C, std=0.02)
            for lora_D in self.loras_D:
                nn.init.zeros_(lora_D)

    def get_gate(self, lora_index):
        """
        为当前模态生成 rank gate；AB 模式同时使用 A、B。
        """
        return rank_gate(self.loras_A[lora_index], self.loras_B[lora_index],
                         mode=self.gate_mode, eps=self.gate_eps,
                         logits=(self.loras_gate_logits[lora_index]
                                 if self.gate_mode in RANK_LEARNED_GATE_MODES else None))

    def forward(self, x: torch.Tensor, lora_index=0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if self.r1 > 0 and self.r2 > 0:
            gate = self.get_gate(lora_index)                  # [r1]
            A = self.loras_A[lora_index] * gate.unsqueeze(1) # [r1, in_features]

            magnitude = F.relu(A.transpose(0, 1) @ self.loras_B[lora_index].transpose(0, 1))
            if self.gate_mode in CHANNEL_GATE_MODES:
                magnitude = calibrate_channel_gate(magnitude, T(self.weight).t(), self.scaling,
                    learned_gate_ratio(self.loras_gate_strengths[lora_index], self.gate_mode),
                    self.gate_mode, output_axis=1,
                    channel_logits=(self.loras_gate_channels[lora_index]
                                    if self.gate_mode in CHANNEL_LEARNED_GATE_MODES else None))
            elif self.gate_mode in CALIBRATED_GATE_MODES or self.gate_mode == 'ab_norm_scalar_bounded':
                ratio = (learned_gate_ratio(self.loras_gate_strengths[lora_index])
                         if self.gate_mode in STRENGTH_LEARNED_GATE_MODES else GATE_UPDATE_RMS_RATIO)
                if self.gate_mode == 'ab_norm_scalar_bounded':
                    ratio = learned_gate_ratio(self.loras_gate_strengths[lora_index], self.gate_mode)
                magnitude = calibrate_gate_magnitude(magnitude, self.weight, self.scaling, ratio)
            direction_matrix = (self.loras_C[lora_index].transpose(0, 1)
                                @ self.loras_D[lora_index].transpose(0, 1))
            dual_update = magnitude * clipped_ste_sign(direction_matrix)
            if self.direction_residual != 0.0:
                dual_update = dual_update + self.direction_residual * direction_matrix
            delta_w = dual_update * self.scaling

            lora_result = self.lora_dropout(x) @ delta_w
            result = result + lora_result

        return result


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlockAllwithGDLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int,
                 num_loras: int, attn_mask: torch.Tensor = None, gate_mode: str = 'ab_norm_out_channel',
                 lora_alpha: float = 1.0, direction_residual: float = 0.0):
        super().__init__()
        self.direction_residual = direction_residual

        self.attn = GDLoRAMultiheadAttention(
            d_model, n_head,
            lora_r1=lora_r1,
            lora_r2=lora_r2,
            num_loras=num_loras,
            lora_alpha=lora_alpha,
            gate_mode=gate_mode,
            direction_residual=direction_residual,
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", GDLoRALinear(d_model, d_model * 4,
                                lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras,
                                lora_alpha=lora_alpha, gate_mode=gate_mode,
                                direction_residual=direction_residual)),
            ("gelu", QuickGELU()),
            ("c_proj", GDLoRALinear(d_model * 4, d_model,
                                  lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras,
                                  lora_alpha=lora_alpha, gate_mode=gate_mode,
                                  direction_residual=direction_residual))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, lora_index):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(
            x, x, x,
            need_weights=False,
            attn_mask=self.attn_mask,
            lora_index=lora_index
        )[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x), lora_index)
        x_mlp = self.mlp[0](self.ln_2(x), lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp, lora_index)
        x = x + x_mlp
        return x


class ResidualAttentionBlockMLPwithGDLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int,
                 num_loras: int, attn_mask: torch.Tensor = None, gate_mode: str = 'ab_norm_out_channel',
                 lora_alpha: float = 1.0, direction_residual: float = 0.0):
        super().__init__()
        self.direction_residual = direction_residual

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", GDLoRALinear(d_model, d_model * 4,
                                lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras,
                                lora_alpha=lora_alpha, gate_mode=gate_mode,
                                direction_residual=direction_residual)),
            ("gelu", QuickGELU()),
            ("c_proj", GDLoRALinear(d_model * 4, d_model,
                                  lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras,
                                  lora_alpha=lora_alpha, gate_mode=gate_mode,
                                  direction_residual=direction_residual))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # 非 LoRA 注意力层全部冻结
        for p in self.attn.parameters():
            p.requires_grad = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x_mlp = self.mlp[0](self.ln_2(x), lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp, lora_index)
        x = x + x_mlp
        return x


class ResidualAttentionBlockwithoutGDLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # 整块冻结
        for p in self.parameters():
            p.requires_grad = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GDLoRATransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        lora_r1: int = 16,
        lora_r2: int = 16,
        num_loras: int = 16,
        lora_layers: int = 2,
        lora_mode: str = 'mlp',
        gate_mode: str = 'ab_norm_out_channel',
        lora_alpha: float = 1.0,
        direction_residual: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.direction_residual = direction_residual
        assert lora_layers <= layers

        if lora_mode == 'all':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutGDLoRA(width, heads, attn_mask)
                 for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockAllwithGDLoRA(
                    width, heads, lora_r1, lora_r2, num_loras, attn_mask,
                    gate_mode=gate_mode, lora_alpha=lora_alpha,
                    direction_residual=direction_residual
                ) for _ in range(lora_layers)]
            )
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutGDLoRA(width, heads, attn_mask)
                 for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithGDLoRA(
                    width, heads, lora_r1, lora_r2, num_loras, attn_mask,
                    gate_mode=gate_mode, lora_alpha=lora_alpha,
                    direction_residual=direction_residual
                ) for _ in range(lora_layers)]
            )
        else:
            raise ValueError("lora_mode must be 'all' or 'mlp'")

        # self.print_trainable_parameters()

    def forward(self, x: torch.Tensor, data_type="RGB"):
        lora_mapping = {
            "RGB": 0, "NIR": 1, "CP": 2, "SK": 3, "TEXT": 4,
            "NIR+CP": 5, "NIR+SK": 6, "NIR+TEXT": 7, "CP+SK": 8, "CP+TEXT": 9, "SK+TEXT": 10,
            "NIR+CP+SK": 11, "NIR+CP+TEXT": 12, "NIR+SK+TEXT": 13, "CP+SK+TEXT": 14,
            "NIR+CP+SK+TEXT": 15
        }
        lora_index = lora_mapping.get(data_type, 0)
        for block in self.resblocks:
            x = block(x, lora_index)
        return x


    def print_trainable_parameters(self):
        total = 0
        trainable = 0
        print("=" * 80)
        print("Trainable parameters:")
        for name, p in self.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
                print(f"[Trainable] {name:80s} {tuple(p.shape)}")
        print("=" * 80)
        print(f"Total params:      {total:,}")
        print(f"Trainable params:  {trainable:,}")
        print(f"Trainable ratio:   {100 * trainable / total:.4f}%")
        print("=" * 80)

class TextGDLoRATransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lora_r1: int=16, lora_r2: int=16, num_loras: int=16, lora_layers:int=2, lora_mode:str='mlp', gate_mode: str='ab_norm_out_channel', lora_alpha: float=1.0, direction_residual: float=0.0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.direction_residual = direction_residual
        assert lora_layers<=layers
        if lora_mode == 'all':
            self.resblocks = nn.ModuleList([ResidualAttentionBlockwithoutGDLoRA(width, heads,attn_mask) for _ in range(layers-lora_layers)]+
                                       [ResidualAttentionBlockAllwithGDLoRA(width, heads, lora_r1, lora_r2, num_loras,attn_mask, gate_mode=gate_mode, lora_alpha=lora_alpha, direction_residual=direction_residual) for _ in range(lora_layers)])
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutGDLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithGDLoRA(width, heads, lora_r1, lora_r2, num_loras, attn_mask, gate_mode=gate_mode, lora_alpha=lora_alpha, direction_residual=direction_residual) for _ in range(lora_layers)])
        else:
            raise("Error")

    def forward(self, x: torch.Tensor, data_type="TEXT"):
        lora_mapping = {"TEXT": 0}
        lora_index = lora_mapping.get(data_type, 0)
        for block in self.resblocks:
            x = block(x, lora_index)
        return x
