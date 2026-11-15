# MIT License for the portions adapted from the credited upstream projects.
#
# Copyright (c) 2021 OpenAI
# Copyright (c) 2023 anosorae
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SCOPE model integration for the HAM-pretrained ViT backbone."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .gd_lora import GDLoRATransformer, TextGDLoRATransformer

class VisionTokenizer(nn.Module):
    def __init__(self, conv1, class_embedding, positional_embedding,ln_pre):
        super(VisionTokenizer, self).__init__()
        self.conv1 = conv1
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.ln_pre = ln_pre

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x

class MultimodalVisionEncoder(nn.Module):
    def __init__(self, transformer, ln_post, proj, modality_aware=True):
        super(MultimodalVisionEncoder, self).__init__()
        self.transformer = transformer
        self.ln_post = ln_post
        self.proj = proj
        self.modality_aware = modality_aware

    def forward(self, x,modality='RGB'):
        x = x.permute(1, 0, 2)
        x = (self.transformer(x, modality) if self.modality_aware
             else self.transformer(x))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x @ self.proj
        return x

class CLIPTextEncoder(nn.Module):
    def __init__(self, token_embedding, positional_embedding,transformer,ln_final,text_projection):
        super(CLIPTextEncoder, self).__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_projection

    def forward(self, text, dtype, modality='TEXT'):
        x = self.token_embedding(text).type(dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.text_projection
        eot_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        embeds = x
        return eot_embed,embeds


class SCOPE(nn.Module):
    """Free-gate Dual-LoRA encoders with optional modality-specific fusion.

    ``backbone`` is an already-loaded HAM-pretrained ViT module. Its visual positional
    embeddings must match the intended image resolution. Alternatively,
    ``from_pretrained`` loads a local HAM checkpoint. Input tokenization and
    image preprocessing are handled by the caller; there is no network access.
    """

    def __init__(self, backbone, rank_magnitude=8, rank_direction=8,
                 alpha=1.0, temperature=0.07):
        super().__init__()
        if min(rank_magnitude, rank_direction) <= 0 or alpha <= 0 or temperature <= 0:
            raise ValueError("Ranks, alpha, and temperature must be positive")
        visual = backbone.visual
        if not hasattr(visual, "transformer"):
            raise ValueError("SCOPE requires a ViT backbone")
        device = next(backbone.parameters()).device
        self.embed_dim = backbone.text_projection.shape[1]
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        adapter = dict(lora_r1=rank_magnitude, lora_r2=rank_direction,
                       gate_mode="ab_norm_out_channel", lora_mode="all",
                       lora_alpha=alpha, direction_residual=0.0)

        # Preserve the encoder initialization order and checkpoint key layout.
        text_blocks = backbone.transformer.resblocks
        text_width = backbone.token_embedding.weight.shape[1]
        text_transformer = TextGDLoRATransformer(
            width=text_width, layers=len(text_blocks),
            heads=text_blocks[0].attn.num_heads,
            attn_mask=copy.deepcopy(text_blocks[0].attn_mask),
            num_loras=1, lora_layers=len(text_blocks), **adapter,
        )
        self._load_backbone(text_transformer, backbone.transformer)
        self.clip_text_encoder = CLIPTextEncoder(
            copy.deepcopy(backbone.token_embedding),
            copy.deepcopy(backbone.positional_embedding), text_transformer,
            copy.deepcopy(backbone.ln_final), copy.deepcopy(backbone.text_projection),
        )
        for modality in ("rgb", "nir", "cp", "sk"):
            tokenizer = VisionTokenizer(
                copy.deepcopy(visual.conv1), copy.deepcopy(visual.class_embedding),
                copy.deepcopy(visual.positional_embedding), copy.deepcopy(visual.ln_pre),
            )
            tokenizer.requires_grad_(True)
            setattr(self, modality + "_tokenizer", tokenizer)
        visual_blocks = visual.transformer.resblocks
        vision_transformer = GDLoRATransformer(
            width=visual.conv1.out_channels, layers=len(visual_blocks),
            heads=visual_blocks[0].attn.num_heads,
            num_loras=4, lora_layers=len(visual_blocks), **adapter,
        )
        self._load_backbone(vision_transformer, visual.transformer)
        self.vision_encoder = MultimodalVisionEncoder(
            vision_transformer, copy.deepcopy(visual.ln_post), copy.deepcopy(visual.proj),
        )
        text_calibration = {id(self.clip_text_encoder.text_projection)}
        for module in self.clip_text_encoder.modules():
            if isinstance(module, nn.LayerNorm):
                text_calibration.update(id(p) for p in module.parameters(recurse=False))
        for name, parameter in self.clip_text_encoder.named_parameters():
            parameter.requires_grad_("lora" in name.lower() or id(parameter) in text_calibration)
        self.vision_encoder.ln_post.requires_grad_(True)
        self.vision_encoder.proj.requires_grad_(True)
        self.float().to(device)

    @classmethod
    def from_pretrained(cls, ham_checkpoint, image_size=(384, 128),
                        rank_magnitude=8, rank_direction=8, alpha=1.0,
                        temperature=0.07):
        """Build SCOPE directly from a local HAM ReID pretrained checkpoint."""
        from .ham import load_ham_backbone
        backbone = load_ham_backbone(ham_checkpoint, image_size=image_size)
        return cls(backbone, rank_magnitude=rank_magnitude,
                   rank_direction=rank_direction, alpha=alpha, temperature=temperature)

    @staticmethod
    def _load_backbone(target, source):
        result = target.load_state_dict(source.state_dict(), strict=False)
        missing = [name for name in result.missing_keys if "lora" not in name.lower()]
        if result.unexpected_keys or missing:
            raise ValueError("Backbone transformer is incompatible with GD-LoRA")

    @property
    def dtype(self):
        return self.rgb_tokenizer.conv1.weight.dtype

    def encode_visual(self, image, modality="rgb"):
        modality = modality.lower()
        if modality not in ("rgb", "nir", "cp", "sk"):
            raise ValueError("Expected rgb, nir, cp, or sk")
        tokens = getattr(self, modality + "_tokenizer")(image.to(dtype=self.dtype))
        return self.vision_encoder(tokens, modality.upper())[:, 0, :]

    def encode_rgb_cls(self, image):
        return self.encode_visual(image, "rgb")

    def encode_nir_cls(self, image):
        return self.encode_visual(image, "nir")

    def encode_cp_cls(self, image):
        return self.encode_visual(image, "cp")

    def encode_sk_cls(self, image):
        return self.encode_visual(image, "sk")

    def encode_text_cls(self, token_ids):
        return self.clip_text_encoder(token_ids, self.dtype)[0]

    def forward(self, inputs):
        """Encode a mapping of preprocessed images and/or CLIP token IDs."""
        if not inputs or any(name not in ("rgb", "nir", "cp", "sk", "text") for name in inputs):
            raise ValueError("Expected a nonempty mapping of supported modalities")
        return {name: getattr(self, "encode_" + name + "_cls")(value)
                for name, value in inputs.items()}

    def _init_mm_adapter(self, heads):
        for adapter in heads:
            for module in adapter.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    nn.init.constant_(module.bias, 0.0)
            nn.init.normal_(adapter[-1].weight, std=1e-3)
            nn.init.constant_(adapter[-1].bias, 0.0)

    def msf_fuse_by_indices(self, modal_features, indices):
        """Return raw fused features, logits, and probabilities in that order.

        ``modal_features`` uses the fixed order [NIR, CP, SK, TEXT]. Unused
        slots may be None. Pass two to four distinct modality indices.
        """
        if not 2 <= len(indices) <= 4 or len(set(indices)) != len(indices):
            raise ValueError("Fusion requires two to four distinct modalities")
        if any(i not in range(4) or modal_features[i] is None for i in indices):
            raise ValueError("A selected modality is missing")
        names = ("mm_two_adapter", "mm_three_adapter", "mm_four_adapter")
        if not hasattr(self, names[len(indices) - 2]):
            raise RuntimeError("Attach and load MSF heads before using learned fusion")
        adapters = getattr(self, names[len(indices) - 2])
        logits = torch.cat([adapters[i](modal_features[i]) for i in indices], dim=1)
        probabilities = F.softmax(logits, dim=1)
        fused = sum(modal_features[i] * probabilities[:, j:j + 1]
                    for j, i in enumerate(indices))
        return fused, logits, probabilities

