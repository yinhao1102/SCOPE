import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class TopSingularValue(nn.Module):
    """
    输入一个 pooled feature: [..., token_dim]
    输出动态 rank 权重: [..., r]
    """
    def __init__(self, token_dim, r, dtype=None, device=None):
        super().__init__()
        self.r = r
        self.token_dim = token_dim
        self.weight = nn.Parameter(
            torch.empty((token_dim, r), dtype=dtype, device=device)
        )
        self.rms_norm = nn.RMSNorm([r])
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode="fan_out")

    def forward(self, x):
        weight = x @ self.weight
        weight = self.rms_norm(weight)
        weight = torch.exp(weight)
        return weight


class DynamicDualLoRAWeight(nn.Module):
    """
    生成“可折叠成矩阵”的动态 LoRA 增量：
    - A/B 分支有动态 rank 权重
    - C/D 分支无动态权重
    - 注意：这里动态权重基于 pooled x，因此一次 forward 内得到的是一个固定增量矩阵
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r1: int,
        r2: int,
        lora_alpha: float = 1.0,
        dtype=None,
        device=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.r1 = r1
        self.r2 = r2
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / math.sqrt(self.r1 * self.r2)

        # A/B branch
        self.A = nn.Parameter(torch.empty((r1, in_features), **factory_kwargs))
        self.B = nn.Parameter(torch.empty((out_features, r1), **factory_kwargs))

        # C/D branch
        self.C = nn.Parameter(torch.empty((r2, in_features), **factory_kwargs))
        self.D = nn.Parameter(torch.empty((out_features, r2), **factory_kwargs))

        self.rank_weight = TopSingularValue(
            token_dim=in_features,
            r=r1,
            dtype=dtype,
            device=device,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.B, std=0.02)
        nn.init.normal_(self.C, std=0.02)
        nn.init.normal_(self.D, std=0.02)

    def pooled_rank_weight(self, x: torch.Tensor):
        """
        x: [L, N, E] or [..., E]
        为了能接入 F.multi_head_attention_forward，
        这里必须生成一次 forward 内共享的 rank 权重。
        """
        if x.dim() == 3:
            # [L, N, E] -> [E]
            pooled = x.mean(dim=(0, 1))
        else:
            pooled = x.mean(dim=0)

        pooled = pooled.to(self.A.dtype)
        rank_weight = self.rank_weight(pooled)  # [r1]
        return rank_weight

    def get_delta_weight(self, x: torch.Tensor):
        """
        返回一个静态矩阵增量 delta_W: [out_features, in_features]

        这里做法：
            AB = B @ diag(g) @ A
            CD = D @ C

        然后保持你原先 dual 形式的思想：
            delta = relu(AB) * sign(CD) * scaling
        """
        g = self.pooled_rank_weight(x)  # [r1]

        # B @ diag(g) @ A
        weighted_A = self.A * g.unsqueeze(1)        # [r1, in_features]
        ab_mat = self.B @ weighted_A                # [out_features, in_features]

        cd_mat = self.D @ self.C                    # [out_features, in_features]

        delta = F.relu(ab_mat) * torch.sign(cd_mat)
        delta = delta * self.scaling
        return delta


class Dynamic_dual_LoRAMultiheadAttention(nn.Module):
    """
    改成用 F.multi_head_attention_forward
    前提：LoRA 增量矩阵必须在一次 forward 内固定，
    所以这里使用 pooled x 生成共享 gate。
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        lora_r1=16,
        lora_r2=16,
        num_loras=16,
        lora_alpha=1,
        lora_dropout=0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = 0.0
        self.lora_alpha = lora_alpha
        self.r1 = lora_r1
        self.r2 = lora_r2

        # base attention params
        self.in_proj_weight = nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.in_proj_bias = nn.Parameter(
            torch.empty((3 * embed_dim,), **factory_kwargs)
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        # dynamic lora weights
        self.in_loras = nn.ModuleList([
            DynamicDualLoRAWeight(
                in_features=embed_dim,
                out_features=3 * embed_dim,
                r1=lora_r1,
                r2=lora_r2,
                lora_alpha=lora_alpha,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_loras)
        ])

        self.out_loras = nn.ModuleList([
            DynamicDualLoRAWeight(
                in_features=embed_dim,
                out_features=embed_dim,
                r1=lora_r1,
                r2=lora_r2,
                lora_alpha=lora_alpha,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_loras)
        ])

        # freeze pretrained weights
        self.in_proj_weight.requires_grad = False
        self.in_proj_bias.requires_grad = False
        self.out_proj.weight.requires_grad = False
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad = False

        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "in_loras"):
            for mod in self.in_loras:
                nn.init.kaiming_uniform_(mod.A, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mod.B, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mod.C, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mod.D, a=math.sqrt(5))
            for mod in self.out_loras:
                nn.init.kaiming_uniform_(mod.A, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mod.B, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mod.C, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mod.D, a=math.sqrt(5))

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
        if not (query is key and key is value):
            raise ValueError("This implementation currently only supports self-attention (query=key=value).")

        x = query
        target_dtype = self.in_proj_weight.dtype
        x_for_lora = x.to(target_dtype)

        delta_in = self.in_loras[lora_index].get_delta_weight(x_for_lora)     # [3E, E]
        delta_out = self.out_loras[lora_index].get_delta_weight(x_for_lora)   # [E, E]

        in_proj_weight = self.in_proj_weight + delta_in
        out_proj_weight = self.out_proj.weight + delta_out

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            use_separate_proj_weight=False,
        )

        return attn_output, attn_output_weights


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
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()


class LoRALinear(nn.Linear, LoRALayer):
    """
    线性层这里仍然保留 token-wise 动态权重是没问题的，
    因为 Linear forward 本来就是逐 token 做的。
    """
    def __init__(
        self,
        self_in_features: int,
        out_features: int,
        lora_r1: int = 16,
        lora_r2: int = 16,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        num_loras: int = 16,
        **kwargs
    ):
        nn.Linear.__init__(self, self_in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r1=lora_r1, r2=lora_r2, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        self.num_loras = num_loras

        if lora_r1 > 0 and lora_r2 > 0:
            self.loras_A = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((lora_r1, self_in_features)))
                for _ in range(num_loras)
            ])
            self.loras_B = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((out_features, lora_r1)))
                for _ in range(num_loras)
            ])
            self.loras_C = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((lora_r2, self_in_features)))
                for _ in range(num_loras)
            ])
            self.loras_D = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((out_features, lora_r2)))
                for _ in range(num_loras)
            ])

            self.lora_rank_weight = nn.ModuleList([
                TopSingularValue(
                    token_dim=self_in_features,
                    r=lora_r1,
                    dtype=self.weight.dtype,
                    device=self.weight.device,
                )
                for _ in range(num_loras)
            ])

            self.scaling = self.lora_alpha / math.sqrt(self.r1 * self.r2)

            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "loras_A"):
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
            x_in = self.lora_dropout(x).to(self.loras_A[lora_index].dtype)

            ab_rank = x_in @ self.loras_A[lora_index].transpose(0, 1)
            rank_weight = self.lora_rank_weight[lora_index](x_in)
            ab_rank = ab_rank * rank_weight
            ab_out = ab_rank @ self.loras_B[lora_index].transpose(0, 1)

            cd_rank = x_in @ self.loras_C[lora_index].transpose(0, 1)
            cd_out = cd_rank @ self.loras_D[lora_index].transpose(0, 1)

            lora_result = F.relu(ab_out) * torch.sign(cd_out)
            lora_result = lora_result * self.scaling
            result = result + lora_result.to(result.dtype)

        return result


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlockAllwithDynamicLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int, num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = Dynamic_dual_LoRAMultiheadAttention(
            d_model,
            n_head,
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
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
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

class ResidualAttentionBlockMLPwithDynamicLoRA(nn.Module):
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
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

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
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MMTransformer_with_duallora_wight(nn.Module):
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
        print('Dual LoRA Mode: {}'.format(lora_mode))

        if lora_mode == 'all':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutDualLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockAllwithDynamicLoRA(width, heads, lora_r1, lora_r2, num_loras, attn_mask) for _ in range(lora_layers)]
            )
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutDualLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithDynamicLoRA(width, heads, lora_r1, lora_r2, num_loras, attn_mask) for _ in range(lora_layers)]
            )
        else:
            raise ValueError("lora_mode must be 'all' or 'mlp'")

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