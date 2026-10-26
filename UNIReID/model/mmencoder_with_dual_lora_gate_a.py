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


def configure_times_new_roman(plt) -> None:
    """Register and select Times New Roman for paper visualizations."""
    from matplotlib import font_manager

    font_path = "/usr/share/fonts/truetype/times.ttf"
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": font_name,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


class Dual_LoRAParameter(nn.Module):
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


class Dual_LoRAMultiheadAttention_gate(nn.Module):
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

        # 包装成 Dual_LoRAParameter
        self.in_proj_weight_withloras = nn.ModuleList([
            Dual_LoRAParameter(
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
            Dual_LoRAParameter(
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
        gate_eps: float = 1e-6,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
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


class ResidualAttentionBlockAllwithDualLoRA_gate(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int,
                 num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = Dual_LoRAMultiheadAttention_gate(
            d_model, n_head,
            lora_r1=lora_r1,
            lora_r2=lora_r2,
            num_loras=num_loras
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4,
                                lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model,
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


class ResidualAttentionBlockMLPwithDualLoRA_gate(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r1: int, lora_r2: int,
                 num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4,
                                lora_r1=lora_r1, lora_r2=lora_r2, num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model,
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


class MMTransformer_with_duallora_gate(nn.Module):
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
                [ResidualAttentionBlockwithoutDualLoRA(width, heads, attn_mask)
                 for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockAllwithDualLoRA_gate(
                    width, heads, lora_r1, lora_r2, num_loras, attn_mask
                ) for _ in range(lora_layers)]
            )
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutDualLoRA(width, heads, attn_mask)
                 for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithDualLoRA_gate(
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

    def visualize_average_lora_gates(
        self,
        save_dir="./gate_visualization",
        lora_indices=None,
        layer_names=("c_fc", "c_proj"),
        cmap="viridis",
        dpi=300,
    ):
        """
        Average rank gates over selected LoRA indices and visualize one heatmap.
        This is more compact for paper figures.
        """
        import os
        import torch
        import matplotlib.pyplot as plt

        configure_times_new_roman(plt)
        os.makedirs(save_dir, exist_ok=True)

        lora_blocks = []
        for block_idx, block in enumerate(self.resblocks):
            if hasattr(block, "mlp") and hasattr(block.mlp[0], "get_gate"):
                lora_blocks.append((block_idx, block))

        if len(lora_blocks) == 0:
            print("No LoRA MLP blocks found.")
            return

        first_lora_layer = lora_blocks[0][1].mlp[0]
        num_loras = len(first_lora_layer.loras_A)
        rank = first_lora_layer.r1

        if lora_indices is None:
            lora_indices = list(range(num_loras))

        name_to_mlp_idx = {
            "c_fc": 0,
            "c_proj": 2,
        }

        for layer_name in layer_names:
            if layer_name not in name_to_mlp_idx:
                print(f"Skip invalid layer_name={layer_name}. Use 'c_fc' or 'c_proj'.")
                continue

            mlp_idx = name_to_mlp_idx[layer_name]
            gate_rows = []
            y_labels = []

            for block_idx, block in lora_blocks:
                lora_layer = block.mlp[mlp_idx]

                gates = []
                for lora_index in lora_indices:
                    if 0 <= lora_index < num_loras:
                        with torch.no_grad():
                            gate = lora_layer.get_gate(lora_index).detach().float().cpu()
                        gates.append(gate)

                if len(gates) == 0:
                    continue

                avg_gate = torch.stack(gates, dim=0).mean(dim=0)  # [rank]
                gate_rows.append(avg_gate)
                y_labels.append(f"L{block_idx}")

            gate_matrix = torch.stack(gate_rows, dim=0).numpy()

            plt.figure(figsize=(max(5, rank * 0.45), max(3, len(lora_blocks) * 0.35)))
            im = plt.imshow(gate_matrix, aspect="auto", cmap=cmap)

            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Rank index")
            plt.ylabel("MLP layer")
            plt.title(f"Average {layer_name} rank gates")

            plt.xticks(range(rank), [str(i + 1) for i in range(rank)])
            plt.yticks(range(len(y_labels)), y_labels)

            plt.tight_layout()

            save_path = os.path.join(
                save_dir,
                f"average_rank_gate_{layer_name}.png"
            )
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()

            print(f"Saved: {save_path}")

    def visualize_single_modal_lora_gates(
        self,
        save_dir="./gate_visualization",
        lora_name_map=None,
        layer_name="c_proj",
        lora_indices=(0, 1, 2, 3),
        cmap="viridis",
        dpi=300,
    ):
        """
        Visualize rank gates for each single-modal LoRA separately.

        Heatmap:
            x-axis: rank index
            y-axis: MLP layer
            value: normalized RMS-based gate value

        Args:
            save_dir: directory to save heatmaps.
            lora_name_map: mapping from lora index to modality name.
            layer_name: "c_fc" or "c_proj".
            lora_indices: LoRA indices to visualize.
            cmap: matplotlib colormap.
            dpi: figure resolution.
        """
        import os
        import math
        import torch
        import matplotlib.pyplot as plt

        configure_times_new_roman(plt)
        os.makedirs(save_dir, exist_ok=True)

        if lora_name_map is None:
            lora_name_map = {
                0: "RGB",
                1: "IR",
                2: "SK",
                3: "CP",
            }

        name_to_mlp_idx = {
            "c_fc": 0,
            "c_proj": 2,
        }

        if layer_name not in name_to_mlp_idx:
            raise ValueError("layer_name must be 'c_fc' or 'c_proj'.")

        mlp_idx = name_to_mlp_idx[layer_name]

        # 收集插入 LoRA 的 MLP block
        lora_blocks = []
        for block_idx, block in enumerate(self.resblocks):
            if hasattr(block, "mlp") and hasattr(block.mlp[mlp_idx], "get_gate"):
                lora_blocks.append((block_idx, block))

        if len(lora_blocks) == 0:
            print("No LoRA MLP blocks found.")
            return

        first_lora_layer = lora_blocks[0][1].mlp[mlp_idx]
        rank = first_lora_layer.r1
        num_loras = len(first_lora_layer.loras_A)

        # 生成 2x2 子图
        n = len(lora_indices)
        ncols = 4
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4.2, nrows * 3.8),
            squeeze=False
        )

        vmin, vmax = None, None
        matrices = []

        # 先收集所有 heatmap matrix，方便统一 colorbar 范围
        for lora_index in lora_indices:
            if lora_index < 0 or lora_index >= num_loras:
                print(f"Skip invalid lora_index={lora_index}.")
                matrices.append(None)
                continue

            rows = []
            for block_idx, block in lora_blocks:
                lora_layer = block.mlp[mlp_idx]
                with torch.no_grad():
                    gate = lora_layer.get_gate(lora_index).detach().float().cpu()
                rows.append(gate*15-14.4)

            mat = torch.stack(rows, dim=0).numpy()
            matrices.append(mat)

        valid_mats = [m for m in matrices if m is not None]
        if len(valid_mats) == 0:
            print("No valid LoRA gates found.")
            return

        vmin = min(m.min() for m in valid_mats)
        vmax = max(m.max() for m in valid_mats)

        last_im = None
        for ax, lora_index, mat in zip(axes.flatten(), lora_indices, matrices):
            if mat is None:
                ax.axis("off")
                continue

            modality_name = lora_name_map.get(lora_index, f"LoRA-{lora_index}")

            last_im = ax.imshow(
                mat,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(f"{modality_name}", fontsize=18, fontweight="bold")
            ax.set_xlabel("Rank Index", fontsize=16)
            if lora_index == 0:
                ax.set_ylabel("MLP layer", fontsize=16)

            ax.set_xticks(range(rank))
            ax.set_xticklabels([str(i + 1) for i in range(rank)], fontsize=13)

            ax.set_yticks(range(len(lora_blocks)))
            ax.set_yticklabels([f"L{idx}" for idx, _ in lora_blocks], fontsize=13)

        # 多余子图关掉
        for ax in axes.flatten()[len(lora_indices):]:
            ax.axis("off")

        # fig.suptitle(f"{layer_name} rank gates of single-modal LoRAs", y=1.02)

        if last_im is not None:
            colorbar = fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                fraction=0.025,
                pad=0.02,
            )
            colorbar.ax.tick_params(labelsize=13)

        save_path = os.path.join(save_dir, f"single_modal_rank_gates_{layer_name}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

class text_Transformer_with_duallora_gate(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lora_r1: int=16, lora_r2: int=16, num_loras: int=16, lora_layers:int=2, lora_mode:str='mlp'):
        super().__init__()
        self.width = width
        self.layers = layers
        assert lora_layers<=layers
        print('Dual LoRA Mode:{}'.format(lora_mode))
        if lora_mode == 'all':
            self.resblocks = nn.ModuleList([ResidualAttentionBlockwithoutDualLoRA(width, heads,attn_mask) for _ in range(layers-lora_layers)]+
                                       [ResidualAttentionBlockAllwithDualLoRA_gate(width, heads, lora_r1, lora_r2, num_loras,attn_mask) for _ in range(lora_layers)])
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutDualLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithDualLoRA_gate(width, heads, lora_r1, lora_r2, num_loras, attn_mask) for _ in range(lora_layers)])
        else:
            raise("Error")

    def forward(self, x: torch.Tensor, data_type="TEXT"):
        lora_mapping = {"TEXT": 0}
        lora_index = lora_mapping.get(data_type, 0)
        for block in self.resblocks:
            x = block(x, lora_index)
        return x
