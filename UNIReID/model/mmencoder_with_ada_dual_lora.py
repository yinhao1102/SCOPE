import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any


# =========================
#   Dual LoRA Core
# =========================
class Dual_LoRAParameter(nn.Module):
    """
    对原始参数矩阵加上 Dual-LoRA 增量：
        ΔW = ReLU(B @ A) * sign(D @ C) * scaling
        W' = W + ΔW

    其中：
        orig_param: [out_features, in_features]
        A: [r1, in_features]
        B: [out_features, r1]
        C: [r2, in_features]
        D: [out_features, r2]
    """
    def __init__(self, orig_param, A_param, B_param, C_param, D_param, lora_alpha, r1, r2):
        super().__init__()
        self.orig_param = orig_param
        self.lora_A_param = A_param
        self.lora_B_param = B_param
        self.lora_C_param = C_param
        self.lora_D_param = D_param
        self.lora_alpha = lora_alpha
        self.r1 = r1
        self.r2 = r2
        self.scaling = self.lora_alpha / math.sqrt(max(self.r1 * self.r2, 1))

        # 动态 rank mask
        self.register_buffer("rank_mask_A", torch.ones(self.r1))
        self.register_buffer("rank_mask_C", torch.ones(self.r2))

    def set_rank_mask(self, mask_A: Optional[torch.Tensor] = None, mask_C: Optional[torch.Tensor] = None):
        if mask_A is not None:
            self.rank_mask_A.copy_(mask_A.to(self.rank_mask_A.device, dtype=self.rank_mask_A.dtype))
        if mask_C is not None:
            self.rank_mask_C.copy_(mask_C.to(self.rank_mask_C.device, dtype=self.rank_mask_C.dtype))

    def combined_matrix(self):
        mask_A = self.rank_mask_A.view(-1, 1).to(dtype=self.lora_A_param.dtype, device=self.lora_A_param.device)
        mask_C = self.rank_mask_C.view(-1, 1).to(dtype=self.lora_C_param.dtype, device=self.lora_C_param.device)

        A = self.lora_A_param * mask_A
        B = self.lora_B_param
        C = self.lora_C_param * mask_C
        D = self.lora_D_param

        lora_contribution = F.relu(B @ A) * torch.sign(D @ C) * self.scaling
        return self.orig_param + lora_contribution.transpose(0, 1) if self.orig_param.shape[0] == lora_contribution.shape[1] else self.orig_param + lora_contribution


# =========================
#   Attention with Dual LoRA
# =========================
class Ada_dual_LoRAMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        lora_r1=16,
        lora_r2=16,
        num_loras=16,
        lora_alpha=1,
        device=None,
        dtype=None,
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
        self.num_loras = num_loras

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        # 正确注册参数
        self.loras_A_in = [nn.Parameter(self.in_proj_weight.new_zeros((self.r1, 3 * embed_dim))) for _ in range(num_loras)]
        self.loras_B_in = [nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, self.r1))) for _ in range(num_loras)]
        self.loras_C_in = [nn.Parameter(self.in_proj_weight.new_zeros((self.r2, 3 * embed_dim))) for _ in range(num_loras)]
        self.loras_D_in = [nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, self.r2))) for _ in range(num_loras)]

        self.loras_A_out = [nn.Parameter(self.out_proj.weight.new_zeros((self.r1, embed_dim))) for _ in range(num_loras)]
        self.loras_B_out = [nn.Parameter(self.out_proj.weight.new_zeros((embed_dim, self.r1))) for _ in range(num_loras)]
        self.loras_C_out = [nn.Parameter(self.out_proj.weight.new_zeros((self.r2, embed_dim))) for _ in range(num_loras)]
        self.loras_D_out = [nn.Parameter(self.out_proj.weight.new_zeros((embed_dim, self.r2))) for _ in range(num_loras)]

        self.scaling = self.lora_alpha / self.r1

        self.in_proj_weight_withloras = nn.ModuleList([
            Dual_LoRAParameter(
                self.in_proj_weight,
                self.loras_A_in[i], self.loras_B_in[i],
                self.loras_C_in[i], self.loras_D_in[i],
                self.lora_alpha, self.r1, self.r2
            )
            for i in range(num_loras)
        ])

        self.out_proj_weight_withloras = nn.ModuleList([
            Dual_LoRAParameter(
                self.out_proj.weight,
                self.loras_A_out[i], self.loras_B_out[i],
                self.loras_C_out[i], self.loras_D_out[i],
                self.lora_alpha, self.r1, self.r2
            )
            for i in range(num_loras)
        ])

        # 冻结原始参数
        self.in_proj_weight.requires_grad = False
        self.in_proj_bias.requires_grad = False
        self.out_proj.weight.requires_grad = False
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad = False

        self.reset_parameters()
        self.bias_k = self.bias_v = None
        self.add_zero_attn = False
        self.dropout = 0.0

        for name, param in self.named_parameters():
            if "loras_" in name:
                param.data = param.data.float()

    def reset_parameters(self):
        if hasattr(self, "loras_A_in"):
            for i in range(self.num_loras):
                nn.init.kaiming_uniform_(self.loras_A_in[i], a=math.sqrt(5))
                nn.init.zeros_(self.loras_B_in[i])
                nn.init.kaiming_uniform_(self.loras_A_out[i], a=math.sqrt(5))
                nn.init.zeros_(self.loras_B_out[i])

                # nn.init.kaiming_uniform_(self.loras_C_in[i], a=math.sqrt(5))
                # nn.init.zeros_(self.loras_D_in[i])
                # nn.init.kaiming_uniform_(self.loras_C_out[i], a=math.sqrt(5))
                # nn.init.zeros_(self.loras_D_out[i])

    def set_lora_rank_mask(
        self,
        lora_index: int,
        in_mask_A: Optional[torch.Tensor] = None,
        in_mask_C: Optional[torch.Tensor] = None,
        out_mask_A: Optional[torch.Tensor] = None,
        out_mask_C: Optional[torch.Tensor] = None,
    ):
        self.in_proj_weight_withloras[lora_index].set_rank_mask(in_mask_A, in_mask_C)
        self.out_proj_weight_withloras[lora_index].set_rank_mask(out_mask_A, out_mask_C)

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
    ):
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


# =========================
#   Linear with Dual LoRA
# =========================
class LoRALayer:
    def __init__(
        self,
        r1: int,
        r2: int,
        lora_alpha: int,
        lora_dropout: float
    ):
        self.r1 = r1
        self.r2 = r2
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class LoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r1: int = 16,
        lora_r2: int = 16,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        num_loras: int = 16,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r1=lora_r1, r2=lora_r2, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        self.num_loras = num_loras

        if lora_r1 > 0 and lora_r2 > 0:
            self.loras_A = nn.ParameterList([nn.Parameter(self.weight.new_zeros((lora_r1, in_features))) for _ in range(num_loras)])
            self.loras_B = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, lora_r1))) for _ in range(num_loras)])
            self.loras_C = nn.ParameterList([nn.Parameter(self.weight.new_zeros((lora_r2, in_features))) for _ in range(num_loras)])
            self.loras_D = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, lora_r2))) for _ in range(num_loras)])

            # 动态 rank mask
            self.register_buffer("rank_mask_A", torch.ones(num_loras, lora_r1))
            self.register_buffer("rank_mask_C", torch.ones(num_loras, lora_r2))

            self.scaling = self.lora_alpha / math.sqrt(max(self.r1 * self.r2, 1))
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def set_rank_mask(self, lora_index: int, mask_A: Optional[torch.Tensor] = None, mask_C: Optional[torch.Tensor] = None):
        if mask_A is not None:
            self.rank_mask_A[lora_index].copy_(mask_A.to(self.rank_mask_A.device, dtype=self.rank_mask_A.dtype))
        if mask_C is not None:
            self.rank_mask_C[lora_index].copy_(mask_C.to(self.rank_mask_C.device, dtype=self.rank_mask_C.dtype))

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'loras_A'):
            for lora_A in self.loras_A:
                nn.init.normal_(lora_A, std=0.02)
            for lora_B in self.loras_B:
                nn.init.normal_(lora_B, std=0.02)
            for lora_C in self.loras_C:
                nn.init.normal_(lora_C, std=0.02)
            for lora_D in self.loras_D:
                nn.init.normal_(lora_D, std=0.02)

    def forward(self, x: torch.Tensor, lora_index=0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if self.r1 > 0 and self.r2 > 0:
            mask_A = self.rank_mask_A[lora_index].view(-1, 1).to(
                dtype=self.loras_A[lora_index].dtype,
                device=self.loras_A[lora_index].device
            )
            mask_C = self.rank_mask_C[lora_index].view(-1, 1).to(
                dtype=self.loras_C[lora_index].dtype,
                device=self.loras_C[lora_index].device
            )

            A = self.loras_A[lora_index] * mask_A
            B = self.loras_B[lora_index]
            C = self.loras_C[lora_index] * mask_C
            D = self.loras_D[lora_index]

            delta_w = F.relu(B @ A) * torch.sign(D @ C)
            lora_result = self.lora_dropout(x) @ delta_w.transpose(0, 1)
            result = result + lora_result * self.scaling

        return result


# =========================
#   Basic Blocks
# =========================
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlockAllwithDualLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int, num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = Ada_dual_LoRAMultiheadAttention(
            d_model, n_head,
            lora_r1=lora_r1,
            lora_r2=lora_r2,
            num_loras=num_loras
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4, lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model, lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, lora_index):
        attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask, lora_index=lora_index)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x), lora_index)
        x_mlp = self.mlp[0](self.ln_2(x), lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp, lora_index)
        x = x + x_mlp
        return x


class ResidualAttentionBlockMLPwithDualLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int, num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4, lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model, lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x_mlp = self.mlp[0](self.ln_2(x), lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp, lora_index)
        x = x + x_mlp
        return x


class ResidualAttentionBlockwithoutDualLoRA(nn.Module):
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

    def attention(self, x: torch.Tensor):
        attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# =========================
#   Main Transformer
# =========================
class MMTransformer_with_adaptive_dual_lora(nn.Module):
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
        lora_mode: str = 'mlp'
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_loras = num_loras
        self.lora_r1 = lora_r1
        self.lora_r2 = lora_r2

        assert lora_layers <= layers
        print('Adaptive Dual LoRA Mode: {}'.format(lora_mode))

        if lora_mode == 'all':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutDualLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockAllwithDualLoRA(width, heads, lora_r1, lora_r2, num_loras, attn_mask) for _ in range(lora_layers)]
            )
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutDualLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithDualLoRA(width, heads, lora_r1, lora_r2, num_loras, attn_mask) for _ in range(lora_layers)]
            )
        else:
            raise ValueError("lora_mode must be 'all' or 'mlp'")

    def get_lora_index(self, data_type="RGB"):
        lora_mapping = {
            "RGB": 0, "NIR": 1, "CP": 2, "SK": 3, "TEXT": 4,
            "NIR+CP": 5, "NIR+SK": 6, "NIR+TEXT": 7, "CP+SK": 8, "CP+TEXT": 9, "SK+TEXT": 10,
            "NIR+CP+SK": 11, "NIR+CP+TEXT": 12, "NIR+SK+TEXT": 13, "CP+SK+TEXT": 14,
            "NIR+CP+SK+TEXT": 15
        }
        return lora_mapping.get(data_type, 0)

    def forward(self, x: torch.Tensor, data_type="RGB"):
        lora_index = self.get_lora_index(data_type)
        for block in self.resblocks:
            x = block(x, lora_index)
        return x


# =========================
#   Dynamic Rank Allocator
# =========================
class AdaptiveDualRankAllocator:
    """
    专门适配 Dual-LoRA 的动态 rank 分配器。

    思路：
    1. 对 A/B 分支按 rank 通道统计重要性
    2. 对 C/D 分支按 rank 通道统计重要性
    3. 每隔 mask_interval 步根据预算更新 rank mask
    4. 初始 warmup 不裁剪，最后 final_warmup 固定 mask 微调
    """
    def __init__(
        self,
        model: nn.Module,
        lora_r1: int,
        lora_r2: int,
        target_r1: int,
        target_r2: int,
        init_warmup: int,
        final_warmup: int,
        mask_interval: int,
        beta1: float,
        beta2: float,
        total_step: Optional[int] = None,
        target_total_r1: Optional[int] = None,
        target_total_r2: Optional[int] = None,
        min_rank_A: int = 1,
        min_rank_C: int = 1,
    ):
        self.model = model
        self.lora_init_r1 = lora_r1
        self.lora_init_r2 = lora_r2
        self.ave_target_r1 = target_r1
        self.ave_target_r2 = target_r2
        self.target_total_r1 = target_total_r1
        self.target_total_r2 = target_total_r2

        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.min_rank_A = min_rank_A
        self.min_rank_C = min_rank_C

        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.rank_pattern = {}
        self.global_step = 0

        assert 0 < beta1 < 1
        assert 0 < beta2 < 1

        self.groups = self._build_groups()
        self.total_rank_A, self.total_rank_C = self._count_total_ranks()

        if self.target_total_r1 is None:
            self.target_total_r1 = self.ave_target_r1 * len(self.groups)
        if self.target_total_r2 is None:
            self.target_total_r2 = self.ave_target_r2 * len(self.groups)

        if self.total_step is not None:
            assert self.total_step > self.initial_warmup + self.final_warmup

    def set_total_step(self, total_step: int):
        self.total_step = total_step
        assert self.total_step > self.initial_warmup + self.final_warmup

    def _build_groups(self):
        groups = []

        for module_name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                for i in range(module.num_loras):
                    groups.append({
                        "name": f"{module_name}.loralinear.{i}",
                        "type": "linear",
                        "module": module,
                        "lora_index": i,
                    })

            elif isinstance(module, Ada_dual_LoRAMultiheadAttention):
                for i in range(module.num_loras):
                    groups.append({
                        "name": f"{module_name}.attn_in.{i}",
                        "type": "attn_in",
                        "module": module,
                        "lora_index": i,
                    })
                    groups.append({
                        "name": f"{module_name}.attn_out.{i}",
                        "type": "attn_out",
                        "module": module,
                        "lora_index": i,
                    })

        return groups

    def _count_total_ranks(self):
        total_A, total_C = 0, 0
        for g in self.groups:
            module = g["module"]
            if isinstance(module, LoRALinear):
                total_A += module.r1
                total_C += module.r2
            elif isinstance(module, Ada_dual_LoRAMultiheadAttention):
                total_A += module.r1
                total_C += module.r2
        return total_A, total_C

    def get_rank_pattern(self):
        return self.rank_pattern

    def _get_group_params(self, group):
        module = group["module"]
        idx = group["lora_index"]
        gtype = group["type"]

        if gtype == "linear":
            A = module.loras_A[idx]
            B = module.loras_B[idx]
            C = module.loras_C[idx]
            D = module.loras_D[idx]
            mask_A = module.rank_mask_A[idx]
            mask_C = module.rank_mask_C[idx]
        elif gtype == "attn_in":
            A = module.loras_A_in[idx]
            B = module.loras_B_in[idx]
            C = module.loras_C_in[idx]
            D = module.loras_D_in[idx]
            mask_A = module.in_proj_weight_withloras[idx].rank_mask_A
            mask_C = module.in_proj_weight_withloras[idx].rank_mask_C
        elif gtype == "attn_out":
            A = module.loras_A_out[idx]
            B = module.loras_B_out[idx]
            C = module.loras_C_out[idx]
            D = module.loras_D_out[idx]
            mask_A = module.out_proj_weight_withloras[idx].rank_mask_A
            mask_C = module.out_proj_weight_withloras[idx].rank_mask_C
        else:
            raise ValueError(f"Unknown group type: {gtype}")

        return A, B, C, D, mask_A, mask_C

    def _update_one_ipt(self, key: str, tensor: torch.Tensor):
        if tensor.grad is None:
            return None

        with torch.no_grad():
            ipt_now = (tensor * tensor.grad).abs().detach()
            if key not in self.ipt:
                self.ipt[key] = torch.zeros_like(ipt_now)
                self.exp_avg_ipt[key] = torch.zeros_like(ipt_now)
                self.exp_avg_unc[key] = torch.zeros_like(ipt_now)

            self.ipt[key] = ipt_now
            self.exp_avg_ipt[key] = self.beta1 * self.exp_avg_ipt[key] + (1 - self.beta1) * ipt_now
            self.exp_avg_unc[key] = self.beta2 * self.exp_avg_unc[key] + (1 - self.beta2) * (ipt_now - self.exp_avg_ipt[key]).abs()

            return self.exp_avg_ipt[key] * self.exp_avg_unc[key]

    def update_ipt(self):
        for g in self.groups:
            base = g["name"]
            A, B, C, D, _, _ = self._get_group_params(g)

            self._update_one_ipt(base + ".A", A)
            self._update_one_ipt(base + ".B", B)
            self._update_one_ipt(base + ".C", C)
            self._update_one_ipt(base + ".D", D)

    def _score_from_ema(self, key):
        if key not in self.exp_avg_ipt:
            return None
        return self.exp_avg_ipt[key] * self.exp_avg_unc[key]

    def _compute_group_scores(self, group):
        """
        返回：
            score_A: [r1]
            score_C: [r2]
        """
        base = group["name"]
        A_score = self._score_from_ema(base + ".A")
        B_score = self._score_from_ema(base + ".B")
        C_score = self._score_from_ema(base + ".C")
        D_score = self._score_from_ema(base + ".D")

        if A_score is None or B_score is None or C_score is None or D_score is None:
            A, B, C, D, mask_A, mask_C = self._get_group_params(group)
            device = A.device
            return torch.ones(A.size(0), device=device), torch.ones(C.size(0), device=device)

        # A: [r1, in], B: [out, r1]
        score_A = A_score.mean(dim=1) + B_score.mean(dim=0)
        # C: [r2, in], D: [out, r2]
        score_C = C_score.mean(dim=1) + D_score.mean(dim=0)

        return score_A, score_C

    def _schedule_budget(self, step: int):
        self.global_step = step

        if step <= self.initial_warmup:
            curr_rank_A = self.total_rank_A
            curr_rank_C = self.total_rank_C
            mask_ind = False
        elif step > self.total_step - self.final_warmup:
            curr_rank_A = self.target_total_r1
            curr_rank_C = self.target_total_r2
            mask_ind = True
        else:
            mul_coeff = 1 - (step - self.initial_warmup) / (self.total_step - self.final_warmup - self.initial_warmup)
            curr_rank_A = self.target_total_r1 + (self.total_rank_A - self.target_total_r1) * (mul_coeff ** 3)
            curr_rank_C = self.target_total_r2 + (self.total_rank_C - self.target_total_r2) * (mul_coeff ** 3)
            curr_rank_A = int(curr_rank_A)
            curr_rank_C = int(curr_rank_C)
            mask_ind = (step % self.mask_interval == 0)

        curr_rank_A = max(0, min(curr_rank_A, self.total_rank_A))
        curr_rank_C = max(0, min(curr_rank_C, self.total_rank_C))
        return curr_rank_A, curr_rank_C, mask_ind

    def _global_threshold(self, all_scores, total_rank, curr_rank):
        if curr_rank >= total_rank:
            return None
        if curr_rank <= 0:
            return float("inf")

        cat_scores = torch.cat(all_scores, dim=0)
        k = total_rank - curr_rank
        if k <= 0:
            return None
        if k >= cat_scores.numel():
            return float("inf")
        return torch.kthvalue(cat_scores, k).values.item()

    def _apply_masks(self, curr_rank_A, curr_rank_C):
        all_scores_A = []
        all_scores_C = []
        group_scores = {}

        for g in self.groups:
            score_A, score_C = self._compute_group_scores(g)
            group_scores[g["name"]] = (score_A, score_C)
            all_scores_A.append(score_A.view(-1))
            all_scores_C.append(score_C.view(-1))

        threshold_A = self._global_threshold(all_scores_A, self.total_rank_A, curr_rank_A)
        threshold_C = self._global_threshold(all_scores_C, self.total_rank_C, curr_rank_C)

        with torch.no_grad():
            total_active_A = 0
            total_active_C = 0

            for g in self.groups:
                name = g["name"]
                module = g["module"]
                idx = g["lora_index"]
                score_A, score_C = group_scores[name]

                if threshold_A is None:
                    mask_A = torch.ones_like(score_A)
                else:
                    mask_A = (score_A > threshold_A).float()

                if threshold_C is None:
                    mask_C = torch.ones_like(score_C)
                else:
                    mask_C = (score_C > threshold_C).float()

                # 至少保留最小 rank
                if mask_A.sum().item() < self.min_rank_A:
                    topk = min(self.min_rank_A, mask_A.numel())
                    top_idx = torch.topk(score_A, k=topk).indices
                    mask_A.zero_()
                    mask_A[top_idx] = 1.0

                if mask_C.sum().item() < self.min_rank_C:
                    topk = min(self.min_rank_C, mask_C.numel())
                    top_idx = torch.topk(score_C, k=topk).indices
                    mask_C.zero_()
                    mask_C[top_idx] = 1.0

                if isinstance(module, LoRALinear):
                    module.set_rank_mask(idx, mask_A, mask_C)
                elif isinstance(module, Ada_dual_LoRAMultiheadAttention):
                    if g["type"] == "attn_in":
                        module.set_lora_rank_mask(
                            idx,
                            in_mask_A=mask_A,
                            in_mask_C=mask_C,
                            out_mask_A=None,
                            out_mask_C=None
                        )
                    elif g["type"] == "attn_out":
                        module.set_lora_rank_mask(
                            idx,
                            in_mask_A=None,
                            in_mask_C=None,
                            out_mask_A=mask_A,
                            out_mask_C=mask_C
                        )

                active_A = int(mask_A.sum().item())
                active_C = int(mask_C.sum().item())
                total_active_A += active_A
                total_active_C += active_C
                self.rank_pattern[name] = {"rank_A": active_A, "rank_C": active_C}

        return threshold_A, threshold_C

    def update_and_mask(self, global_step: int):
        if self.total_step is None:
            raise ValueError("Please set total_step before calling update_and_mask().")

        if global_step < self.total_step - self.final_warmup:
            self.update_ipt()

        curr_rank_A, curr_rank_C, mask_ind = self._schedule_budget(global_step)

        if mask_ind:
            threshold_A, threshold_C = self._apply_masks(curr_rank_A, curr_rank_C)
        else:
            threshold_A, threshold_C = None, None

        return {
            "curr_rank_A": curr_rank_A,
            "curr_rank_C": curr_rank_C,
            "threshold_A": threshold_A,
            "threshold_C": threshold_C,
            "mask_updated": mask_ind,
        }


# =========================
#   Optional Orth Regu
# =========================
def compute_dual_lora_orth_regu(model, regu_weight=0.1):
    """
    对 Dual-LoRA 参数做简单正交正则：
        A A^T ≈ I
        B^T B ≈ I
        C C^T ≈ I
        D^T D ≈ I
    """
    regu_loss = 0.0
    num_param = 0

    for module in model.modules():
        if isinstance(module, LoRALinear):
            for i in range(module.num_loras):
                for p, is_A_like in [
                    (module.loras_A[i], True),
                    (module.loras_B[i], False),
                    (module.loras_C[i], True),
                    (module.loras_D[i], False),
                ]:
                    para_cov = p @ p.T if is_A_like else p.T @ p
                    I = torch.eye(para_cov.size(0), device=para_cov.device, dtype=para_cov.dtype)
                    regu_loss = regu_loss + torch.norm(para_cov - I, p="fro")
                    num_param += 1

        elif isinstance(module, Ada_dual_LoRAMultiheadAttention):
            for i in range(module.num_loras):
                for p, is_A_like in [
                    (module.loras_A_in[i], True),
                    (module.loras_B_in[i], False),
                    (module.loras_C_in[i], True),
                    (module.loras_D_in[i], False),
                    (module.loras_A_out[i], True),
                    (module.loras_B_out[i], False),
                    (module.loras_C_out[i], True),
                    (module.loras_D_out[i], False),
                ]:
                    para_cov = p @ p.T if is_A_like else p.T @ p
                    I = torch.eye(para_cov.size(0), device=para_cov.device, dtype=para_cov.dtype)
                    regu_loss = regu_loss + torch.norm(para_cov - I, p="fro")
                    num_param += 1

    if num_param == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return regu_weight * regu_loss / num_param