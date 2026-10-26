import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math
from typing import Optional

class Dual_LoRAParameter(nn.Module):
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
        self.scaling = self.lora_alpha / math.sqrt(self.r1 * self.r2)

        # 给 A 的每个 rank 一个标量权重
        self.weight_A = nn.Parameter(self.orig_param.new_ones((self.r1, 1)))
        self.rms_norm = nn.RMSNorm([r1])

    def update_weight_A(self):
        """
        根据 A_param 生成新的 rank 权重
        假设 A_param shape = [r1, in_features]
        """
        with torch.no_grad():
            # 先转置到 [in_features, r1]，让最后一维等于 r1，才能做 RMSNorm
            w = self.rms_norm(self.lora_A_param.T)   # [in_features, r1]
            w = torch.exp(w)                         # [in_features, r1]
            w = w.mean(dim=0, keepdim=True).T       # [r1, 1]
            self.weight_A.copy_(w)

    def combined_matrix(self):
        """
        组合原始参数矩阵和 LoRA 矩阵，得到一个合并后的矩阵
        """
        # self.update_weight_A()  # 更新 A 的权重
        A_weighted = self.weight_A * self.lora_A_param   # [r1, in_features]

        lora_contribution = (
            F.relu(A_weighted.T @ self.lora_B_param.T)
            * torch.sign(self.lora_C_param.T @ self.lora_D_param.T)
            * self.scaling
        )
        return self.orig_param + lora_contribution


class Dual_LoRAMultiheadAttention_A(nn.Module):
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
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lora_alpha = lora_alpha
        self.r1 = lora_r1
        self.r2 = lora_r2

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        self.loras_A_in = [nn.Parameter(self.in_proj_weight.new_zeros((self.r1, 3 * embed_dim))) for _ in range(num_loras)]
        self.loras_B_in = [nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, self.r1))) for _ in range(num_loras)]
        self.loras_C_in = [nn.Parameter(self.in_proj_weight.new_zeros((self.r2, 3 * embed_dim))) for _ in range(num_loras)]
        self.loras_D_in = [nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, self.r2))) for _ in range(num_loras)]

        self.loras_A_out = [nn.Parameter(self.out_proj.weight.new_zeros((self.r1, embed_dim))) for _ in range(num_loras)]
        self.loras_B_out = [nn.Parameter(self.out_proj.weight.new_zeros((embed_dim, self.r1))) for _ in range(num_loras)]
        self.loras_C_out = [nn.Parameter(self.out_proj.weight.new_zeros((self.r2, embed_dim))) for _ in range(num_loras)]
        self.loras_D_out = [nn.Parameter(self.out_proj.weight.new_zeros((embed_dim, self.r2))) for _ in range(num_loras)]

        self.scaling = self.lora_alpha / self.r1
            # Freezing the pre-trained weight matrix

        self.in_proj_weight_withloras = nn.ModuleList([Dual_LoRAParameter(self.in_proj_weight,self.loras_A_in[i],self.loras_B_in[i],self.loras_C_in[i],self.loras_D_in[i],self.lora_alpha,self.r1,self.r2) for i in range(num_loras)])
        self.out_proj_weight_withloras = nn.ModuleList([Dual_LoRAParameter(self.out_proj.weight,self.loras_A_out[i],self.loras_B_out[i],self.loras_C_out[i],self.loras_D_out[i],self.lora_alpha,self.r1,self.r2) for i in range(num_loras)])

        self.in_proj_weight.requires_grad = False
        self.in_proj_bias.requires_grad = False
        self.out_proj.requires_grad = False

        self.reset_parameters()
        self.bias_k = self.bias_v = None
        self.add_zero_attn = False
        self.dropout = 0.0

        for name, param in self.named_parameters():
            if "lora" in name:
                param.data = param.data.float()

    def reset_parameters(self):
        if hasattr(self, 'loras_A_in'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            for i in range(len(self.loras_A_in)):
                nn.init.kaiming_uniform_(self.loras_A_in[i], a=math.sqrt(5))
                nn.init.zeros_(self.loras_B_in[i])
                nn.init.kaiming_uniform_(self.loras_A_out[i], a=math.sqrt(5))
                nn.init.zeros_(self.loras_B_out[i])

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

        ## 在合并权重之前，更新A的权重
        self.in_proj_weight_withloras[lora_index].update_weight_A()
        self.out_proj_weight_withloras[lora_index].update_weight_A()

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


# 定义 LoRA 模块
class LoRALayer():
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
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r1: int = 16,
        lora_r2: int = 16,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        num_loras: int = 16,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r1=lora_r1, r2=lora_r2, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if lora_r1 > 0 and lora_r2 > 0:
            self.loras_A = nn.ParameterList([nn.Parameter(self.weight.new_zeros((lora_r1, in_features))) for _ in range(num_loras)])
            self.loras_B = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, lora_r1))) for _ in range(num_loras)])
            self.loras_C = nn.ParameterList([nn.Parameter(self.weight.new_zeros((lora_r2, in_features))) for _ in range(num_loras)])
            self.loras_D = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, lora_r2))) for _ in range(num_loras)])
            self.scaling = self.lora_alpha / math.sqrt(self.r1 * self.r2)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'loras_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
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
        if self.r1 > 0 and self.r2 > 0:
            result = F.linear(x, T(self.weight), bias=self.bias)
            lora_result = (self.lora_dropout(x) @ (F.relu(self.loras_A[lora_index].transpose(0, 1) @ self.loras_B[lora_index].transpose(0, 1))*torch.sign(self.loras_C[lora_index].transpose(0, 1) @ self.loras_D[lora_index].transpose(0, 1)))) * self.scaling
            result = result + lora_result
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlockAllwithDualLoRA_A(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int, num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = Dual_LoRAMultiheadAttention_A(d_model, n_head,lora_r1=lora_r1,lora_r2=lora_r2,num_loras=num_loras)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4,lora_r1=lora_r1, lora_r2=lora_r2,num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model,lora_r1=lora_r1, lora_r2=lora_r2,num_loras=num_loras))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor,lora_index):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask,lora_index=lora_index)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x),lora_index)
        x_mlp = self.mlp[0](self.ln_2(x),lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp,lora_index)
        x = x + x_mlp
        return x


class ResidualAttentionBlockMLPwithDualLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int, num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4,lora_r1=lora_r1,lora_r2=lora_r2,num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model,lora_r1=lora_r1,lora_r2=lora_r2,num_loras=num_loras))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x_mlp = self.mlp[0](self.ln_2(x),lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp,lora_index)
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

    def forward(self, x: torch.Tensor,lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MMTransformer_with_duallora_A(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lora_r1: int=16, lora_r2: int=16, num_loras: int=16, lora_layers:int=2, lora_mode:str='mlp'):
        super().__init__()
        self.width = width
        self.layers = layers
        assert lora_layers<=layers
        print('Dual LoRA Mode:{}'.format(lora_mode))
        if lora_mode == 'all':
            self.resblocks = nn.ModuleList([ResidualAttentionBlockwithoutDualLoRA(width, heads,attn_mask) for _ in range(layers-lora_layers)]+
                                       [ResidualAttentionBlockAllwithDualLoRA_A(width, heads, lora_r1, lora_r2, num_loras,attn_mask) for _ in range(lora_layers)])
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutDualLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithDualLoRA(width, heads, lora_r1, lora_r2, num_loras, attn_mask) for _ in range(lora_layers)])
        else:
            raise("Error")

    def forward(self, x: torch.Tensor, data_type="RGB"):
        lora_mapping = {"RGB": 0, "NIR": 1, "CP": 2, "SK": 3, "TEXT": 4,
                        "NIR+CP":5,"NIR+SK":6,"NIR+TEXT":7,"CP+SK":8,"CP+TEXT":9,"SK+TEXT":10,
                        "NIR+CP+SK":11,"NIR+CP+TEXT":12,"NIR+SK+TEXT":13,"CP+SK+TEXT":14,
                        "NIR+CP+SK+TEXT":15}
        lora_index = lora_mapping.get(data_type, 0)
        for block in self.resblocks:
            x = block(x, lora_index)
        return x