"""Modality-specific DoRA layers for the UNIReID ablation.

The weight update follows NVlabs/DoRA: the pretrained weight direction is
adapted with a low-rank update while a separate vector learns its magnitude.
"""

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weight_norm(weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.linalg.vector_norm(weight.float(), dim=1).clamp_min(eps)


def _dora_weight(
    weight: torch.Tensor,
    dora_A: torch.Tensor,
    dora_B: torch.Tensor,
    magnitude: torch.Tensor,
    scaling: float,
    eps: float,
) -> torch.Tensor:
    direction = weight.float() + (dora_B.float() @ dora_A.float()) * scaling
    # This is the memory-saving "simple DoRA" normalization from NVlabs/DoRA.
    direction_norm = _weight_norm(direction, eps).detach()
    normalized = direction * (magnitude.float() / direction_norm).unsqueeze(1)
    return normalized.to(weight.dtype)


class DoRAMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dora_r: int = 16,
        num_doras: int = 16,
        dora_alpha: float = 1.0,
        device=None,
        dtype=None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if dora_r <= 0:
            raise ValueError("dora_r must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dora_r = dora_r
        self.scaling = dora_alpha / dora_r
        self.eps = eps

        self.in_proj_weight = nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.in_proj_bias = nn.Parameter(
            torch.empty(3 * embed_dim, **factory_kwargs)
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        self.dora_A_in = nn.ParameterList([
            nn.Parameter(torch.empty((dora_r, embed_dim), **factory_kwargs))
            for _ in range(num_doras)
        ])
        self.dora_B_in = nn.ParameterList([
            nn.Parameter(torch.empty((3 * embed_dim, dora_r), **factory_kwargs))
            for _ in range(num_doras)
        ])
        self.dora_magnitude_in = nn.ParameterList([
            nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
            for _ in range(num_doras)
        ])

        self.dora_A_out = nn.ParameterList([
            nn.Parameter(torch.empty((dora_r, embed_dim), **factory_kwargs))
            for _ in range(num_doras)
        ])
        self.dora_B_out = nn.ParameterList([
            nn.Parameter(torch.empty((embed_dim, dora_r), **factory_kwargs))
            for _ in range(num_doras)
        ])
        self.dora_magnitude_out = nn.ParameterList([
            nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
            for _ in range(num_doras)
        ])

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

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        for dora_A, dora_B in zip(self.dora_A_in, self.dora_B_in):
            nn.init.kaiming_uniform_(dora_A, a=math.sqrt(5))
            nn.init.zeros_(dora_B)
        for dora_A, dora_B in zip(self.dora_A_out, self.dora_B_out):
            nn.init.kaiming_uniform_(dora_A, a=math.sqrt(5))
            nn.init.zeros_(dora_B)
        self.initialize_dora_magnitudes()

    @torch.no_grad()
    def initialize_dora_magnitudes(self) -> None:
        in_norm = _weight_norm(self.in_proj_weight, self.eps)
        out_norm = _weight_norm(self.out_proj.weight, self.eps)
        for magnitude in self.dora_magnitude_in:
            magnitude.copy_(in_norm.to(magnitude.dtype))
        for magnitude in self.dora_magnitude_out:
            magnitude.copy_(out_norm.to(magnitude.dtype))

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
        lora_index: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not 0 <= lora_index < len(self.dora_A_in):
            raise IndexError(f"DoRA index {lora_index} is out of range")

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        in_proj_weight = _dora_weight(
            self.in_proj_weight,
            self.dora_A_in[lora_index],
            self.dora_B_in[lora_index],
            self.dora_magnitude_in[lora_index],
            self.scaling,
            self.eps,
        )
        out_proj_weight = _dora_weight(
            self.out_proj.weight,
            self.dora_A_out[lora_index],
            self.dora_B_out[lora_index],
            self.dora_magnitude_out[lora_index],
            self.scaling,
            self.eps,
        )

        return F.multi_head_attention_forward(
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


class DoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dora_r: int = 16,
        dora_alpha: float = 1.0,
        dora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        num_doras: int = 16,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        if dora_r <= 0:
            raise ValueError("dora_r must be positive")
        super().__init__(in_features, out_features, **kwargs)
        self.dora_r = dora_r
        self.scaling = dora_alpha / dora_r
        self.fan_in_fan_out = fan_in_fan_out
        self.eps = eps
        self.dora_dropout = (
            nn.Dropout(p=dora_dropout) if dora_dropout > 0.0 else nn.Identity()
        )

        self.dora_A = nn.ParameterList([
            nn.Parameter(self.weight.new_empty((dora_r, in_features)))
            for _ in range(num_doras)
        ])
        self.dora_B = nn.ParameterList([
            nn.Parameter(self.weight.new_empty((out_features, dora_r)))
            for _ in range(num_doras)
        ])
        self.dora_magnitude = nn.ParameterList([
            nn.Parameter(self.weight.new_empty(out_features))
            for _ in range(num_doras)
        ])

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        for dora_A, dora_B in zip(self.dora_A, self.dora_B):
            nn.init.kaiming_uniform_(dora_A, a=math.sqrt(5))
            nn.init.zeros_(dora_B)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        self.initialize_dora_magnitudes()

    def _base_weight(self) -> torch.Tensor:
        return self.weight.transpose(0, 1) if self.fan_in_fan_out else self.weight

    @torch.no_grad()
    def initialize_dora_magnitudes(self) -> None:
        norm = _weight_norm(self._base_weight(), self.eps)
        for magnitude in self.dora_magnitude:
            magnitude.copy_(norm.to(magnitude.dtype))

    def forward(self, x: torch.Tensor, lora_index: int = 0) -> torch.Tensor:
        if not 0 <= lora_index < len(self.dora_A):
            raise IndexError(f"DoRA index {lora_index} is out of range")

        base_weight = self._base_weight()
        dora_A = self.dora_A[lora_index]
        dora_B = self.dora_B[lora_index]
        magnitude = self.dora_magnitude[lora_index]
        delta = (dora_B.float() @ dora_A.float()) * self.scaling
        direction = base_weight.float() + delta
        direction_norm = _weight_norm(direction, self.eps).detach()
        norm_scale = (magnitude.float() / direction_norm).to(x.dtype)

        adapter_input = self.dora_dropout(x)
        base_result = F.linear(x, base_weight, bias=None)
        base_adjustment = F.linear(adapter_input, base_weight, bias=None)
        dora_result = F.linear(adapter_input, delta.to(base_weight.dtype), bias=None)
        result = base_result + (norm_scale - 1).unsqueeze(0) * base_adjustment
        result = result + norm_scale.unsqueeze(0) * dora_result
        if self.bias is not None:
            result = result + self.bias
        return result


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        return super().forward(x.float()).to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlockAllWithDoRA(nn.Module):
    def __init__(self, d_model, n_head, dora_r, num_doras, dora_alpha, dora_dropout, attn_mask=None):
        super().__init__()
        self.attn = DoRAMultiheadAttention(
            d_model, n_head, dora_r=dora_r, num_doras=num_doras,
            dora_alpha=dora_alpha,
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", DoRALinear(
                d_model, d_model * 4, dora_r=dora_r, num_doras=num_doras,
                dora_alpha=dora_alpha, dora_dropout=dora_dropout,
            )),
            ("gelu", QuickGELU()),
            ("c_proj", DoRALinear(
                d_model * 4, d_model, dora_r=dora_r, num_doras=num_doras,
                dora_alpha=dora_alpha, dora_dropout=dora_dropout,
            )),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, lora_index):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask, lora_index=lora_index)[0]

    def forward(self, x, lora_index=0):
        x = x + self.attention(self.ln_1(x), lora_index)
        x_mlp = self.mlp[0](self.ln_2(x), lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x = x + self.mlp[2](x_mlp, lora_index)
        return x


class ResidualAttentionBlockMLPWithDoRA(nn.Module):
    def __init__(self, d_model, n_head, dora_r, num_doras, dora_alpha, dora_dropout, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", DoRALinear(
                d_model, d_model * 4, dora_r=dora_r, num_doras=num_doras,
                dora_alpha=dora_alpha, dora_dropout=dora_dropout,
            )),
            ("gelu", QuickGELU()),
            ("c_proj", DoRALinear(
                d_model * 4, d_model, dora_r=dora_r, num_doras=num_doras,
                dora_alpha=dora_alpha, dora_dropout=dora_dropout,
            )),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        for parameter in self.attn.parameters():
            parameter.requires_grad = False

    def attention(self, x):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x_mlp = self.mlp[0](self.ln_2(x), lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x = x + self.mlp[2](x_mlp, lora_index)
        return x


class ResidualAttentionBlockWithoutDoRA(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        for parameter in self.parameters():
            parameter.requires_grad = False

    def attention(self, x):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class MMTransformerWithDoRA(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        dora_r: int = 16,
        num_doras: int = 16,
        dora_layers: int = 2,
        dora_mode: str = "mlp",
        dora_alpha: float = 1.0,
        dora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dora_layers > layers:
            raise ValueError("dora_layers cannot exceed transformer layers")
        self.width = width
        self.layers = layers
        print(f"DoRA mode: {dora_mode}, rank: {dora_r}, alpha: {dora_alpha}")

        frozen = [
            ResidualAttentionBlockWithoutDoRA(width, heads, attn_mask)
            for _ in range(layers - dora_layers)
        ]
        block_cls = {
            "all": ResidualAttentionBlockAllWithDoRA,
            "mlp": ResidualAttentionBlockMLPWithDoRA,
        }.get(dora_mode)
        if block_cls is None:
            raise ValueError("dora_mode must be 'all' or 'mlp'")
        adapted = [
            block_cls(
                width, heads, dora_r, num_doras, dora_alpha,
                dora_dropout, attn_mask,
            )
            for _ in range(dora_layers)
        ]
        self.resblocks = nn.ModuleList(frozen + adapted)

    @torch.no_grad()
    def initialize_dora_magnitudes(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "initialize_dora_magnitudes"):
                module.initialize_dora_magnitudes()

    def forward(self, x: torch.Tensor, data_type: str = "RGB") -> torch.Tensor:
        dora_mapping = {
            "RGB": 0, "NIR": 1, "CP": 2, "SK": 3, "TEXT": 4,
            "NIR+CP": 5, "NIR+SK": 6, "NIR+TEXT": 7,
            "CP+SK": 8, "CP+TEXT": 9, "SK+TEXT": 10,
            "NIR+CP+SK": 11, "NIR+CP+TEXT": 12,
            "NIR+SK+TEXT": 13, "CP+SK+TEXT": 14,
            "NIR+CP+SK+TEXT": 15,
        }
        dora_index = dora_mapping.get(data_type, 0)
        for block in self.resblocks:
            x = block(x, dora_index)
        return x


class TextTransformerWithDoRA(MMTransformerWithDoRA):
    def forward(self, x: torch.Tensor, data_type: str = "TEXT") -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, 0)
        return x
