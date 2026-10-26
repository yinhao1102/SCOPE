import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math
from typing import Optional


def clipped_ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Hard sign in forward, clipped straight-through gradient in backward."""
    clipped = x.clamp(-1.0, 1.0)
    return x.sign().detach() + clipped - clipped.detach()


class GDLoRAParameter(nn.Module):
    def __init__(self, orig_param, A_param, B_param, C_param, D_param,
                 lora_alpha, r1, r2, gate_eps=1e-6):
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

    def get_gate_A_rms(self):
        """
        根据 A_param 自适应生成 gate，不引入额外可学习参数
        A: [r1, in_features]
        返回:
            gate: [r1]
        """
        # 每个 rank 行的 RMS 大小
        gate = torch.sqrt(torch.mean(self.lora_A_param ** 2, dim=1) + self.gate_eps)  # [r1]

        # 均值归一化，避免整体缩放漂移
        gate = gate / (gate.mean() + self.gate_eps)
        ## 查看gate的分布情况
        # print("Gate A stats: mean {:.4f}, std {:.4f}, min {:.4f}, max {:.4f}".format(gate.mean().item(), gate.std().item(), gate.min().item(), gate.max().item()))

        return gate

    def combined_matrix(self):
        """
        组合原始参数矩阵和 LoRA 矩阵，得到一个合并后的矩阵
        """
        gate_A = self.get_gate_A_rms()                      # [r1]
        A = self.lora_A_param * gate_A.unsqueeze(1)        # [r1, in_features]

        lora_contribution = (
            F.relu(A.T @ self.lora_B_param.T) *
            clipped_ste_sign(self.lora_C_param.T @ self.lora_D_param.T)
        ) * self.scaling

        return self.orig_param + lora_contribution


class GDLoRAMultiheadAttention(nn.Module):
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
        gate_eps=1e-6,
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
                gate_eps=self.gate_eps
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
                gate_eps=self.gate_eps
            ) for i in range(num_loras)
        ])

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


class GDLoRALinear(nn.Linear, GDLoRALayer):
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
        gate_eps: float = 1e-6,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GDLoRALayer.__init__(
            self,
            r1=lora_r1,
            r2=lora_r2,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.gate_eps = gate_eps

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
        根据当前 LoRA A 参数自适应生成 gate
        """
        A = self.loras_A[lora_index]  # [r1, in_features]
        gate = torch.sqrt(torch.mean(A ** 2, dim=1) + self.gate_eps)  # [r1]
        gate = gate / (gate.mean() + self.gate_eps)
        return gate

    def forward(self, x: torch.Tensor, lora_index=0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if self.r1 > 0 and self.r2 > 0:
            gate = self.get_gate(lora_index)                  # [r1]
            A = self.loras_A[lora_index] * gate.unsqueeze(1) # [r1, in_features]

            delta_w = (
                F.relu(A.transpose(0, 1) @ self.loras_B[lora_index].transpose(0, 1)) *
                clipped_ste_sign(
                    self.loras_C[lora_index].transpose(0, 1)
                    @ self.loras_D[lora_index].transpose(0, 1)
                )
            ) * self.scaling

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
                 num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = GDLoRAMultiheadAttention(
            d_model, n_head,
            lora_r1=lora_r1,
            lora_r2=lora_r2,
            num_loras=num_loras
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", GDLoRALinear(d_model, d_model * 4,
                                lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", GDLoRALinear(d_model * 4, d_model,
                                  lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras))
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
                 num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", GDLoRALinear(d_model, d_model * 4,
                                lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", GDLoRALinear(d_model * 4, d_model,
                                  lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras))
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
        lora_mode: str = 'mlp'
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        assert lora_layers <= layers
        print('GD-LoRA mode: {}'.format(lora_mode))

        if lora_mode == 'all':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutGDLoRA(width, heads, attn_mask)
                 for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockAllwithGDLoRA(
                    width, heads, lora_r1, lora_r2, num_loras, attn_mask
                ) for _ in range(lora_layers)]
            )
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutGDLoRA(width, heads, attn_mask)
                 for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithGDLoRA(
                    width, heads, lora_r1, lora_r2, num_loras, attn_mask
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
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lora_r1: int=16, lora_r2: int=16, num_loras: int=16, lora_layers:int=2, lora_mode:str='mlp'):
        super().__init__()
        self.width = width
        self.layers = layers
        assert lora_layers<=layers
        print('GD-LoRA text mode: {}'.format(lora_mode))
        if lora_mode == 'all':
            self.resblocks = nn.ModuleList([ResidualAttentionBlockwithoutGDLoRA(width, heads,attn_mask) for _ in range(layers-lora_layers)]+
                                       [ResidualAttentionBlockAllwithGDLoRA(width, heads, lora_r1, lora_r2, num_loras,attn_mask) for _ in range(lora_layers)])
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutGDLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithGDLoRA(width, heads, lora_r1, lora_r2, num_loras, attn_mask) for _ in range(lora_layers)])
        else:
            raise("Error")

    def forward(self, x: torch.Tensor, data_type="TEXT"):
        lora_mapping = {"TEXT": 0}
        lora_index = lora_mapping.get(data_type, 0)
        for block in self.resblocks:
            x = block(x, lora_index)
        return x
