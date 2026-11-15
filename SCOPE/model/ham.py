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

"""Load the HAM ReID-pretrained ViT backbone from a local tensor checkpoint.

Transformer blocks follow OpenAI CLIP and IRRA.
No checkpoint is downloaded and no training metadata is retained.
"""
from collections import OrderedDict
from collections.abc import Mapping
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .gd_lora import LayerNorm, QuickGELU


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, 4 * width)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(4 * width, width)),
        ]))
        self.ln_2 = LayerNorm(width)
        self.attn_mask = attn_mask

    def forward(self, x):
        mask = self.attn_mask.to(x.device, x.dtype) if self.attn_mask is not None else None
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width, layers, attn_mask=None):
        super().__init__()
        if width % 64:
            raise ValueError("HAM transformer widths must be divisible by 64")
        self.width, self.layers = width, layers
        self.resblocks = nn.Sequential(*[
            AttentionBlock(width, width // 64, attn_mask) for _ in range(layers)
        ])

    def forward(self, x):
        return self.resblocks(x)


class VisionBackbone(nn.Module):
    def __init__(self, width, layers, patch_size, image_size, embed_dim):
        super().__init__()
        self.input_resolution = tuple(image_size)
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.empty(width))
        self.grid = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.positional_embedding = nn.Parameter(torch.empty(1 + math.prod(self.grid), width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(torch.empty(width, embed_dim))


class HAMBackbone(nn.Module):
    """Checkpoint-compatible container used to initialize the SCOPE encoders."""
    def __init__(self, state, image_size):
        super().__init__()
        vision_width, _, patch_size, other_patch = state["visual.conv1.weight"].shape
        if patch_size != other_patch:
            raise ValueError("Expected square ViT patches")
        if len(image_size) != 2 or any(v < patch_size or v % patch_size for v in image_size):
            raise ValueError("Image dimensions must be positive multiples of the patch size")
        embed_dim = state["text_projection"].shape[1]
        vocab_size, text_width = state["token_embedding.weight"].shape
        context_length = state["positional_embedding"].shape[0]
        vision_layers = _layer_count(state, "visual.transformer.resblocks.")
        text_layers = _layer_count(state, "transformer.resblocks.")
        self.visual = VisionBackbone(vision_width, vision_layers, patch_size, image_size, embed_dim)
        mask = torch.full((context_length, context_length), float("-inf")).triu_(1)
        self.transformer = Transformer(text_width, text_layers, mask)
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, text_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, text_width))
        self.ln_final = LayerNorm(text_width)
        self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))


def _layer_count(state, prefix):
    indices = {int(key[len(prefix):].split(".", 1)[0])
               for key in state if key.startswith(prefix)}
    if not indices or indices != set(range(len(indices))):
        raise ValueError("Incomplete HAM transformer checkpoint")
    return len(indices)


def _tensor_state(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a HAM tensor state dictionary")
    for wrapper in ("model", "state_dict"):
        if wrapper in checkpoint and isinstance(checkpoint[wrapper], Mapping):
            checkpoint = checkpoint[wrapper]
            break
    state = {key: value for key, value in checkpoint.items() if isinstance(value, torch.Tensor)}
    for prefix in ("module.base_model.", "base_model.", "module."):
        if any(key.startswith(prefix + "visual.") for key in state):
            state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            break
    required = ("visual.conv1.weight", "visual.proj", "visual.positional_embedding",
                "token_embedding.weight", "positional_embedding", "text_projection")
    if any(key not in state for key in required):
        raise ValueError("Expected HAM ReID pretrained weights, not a SCOPE adapter checkpoint")
    for key in ("logit_scale", "input_resolution", "context_length", "vocab_size"):
        state.pop(key, None)
    return state


def _resize_position_embedding(value, target_grid):
    count = value.shape[0] - 1
    if count == math.prod(target_grid):
        return value
    side = math.isqrt(count)
    if side * side == count:
        source_grid = (side, side)
    elif count == 24 * 8:
        source_grid = (24, 8)  # HAM's 384 x 128 images with 16 x 16 patches.
    else:
        raise ValueError("Unrecognized HAM positional grid; use its original image resolution")
    patches = value[1:].float().reshape(1, *source_grid, -1).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=target_grid, mode="bilinear", align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(math.prod(target_grid), -1)
    return torch.cat((value[:1].float(), patches), dim=0).to(value.dtype)


def load_ham_backbone(checkpoint_path, image_size=(384, 128)):
    """Read official HAM weights, strip their wrapper, and validate every tensor."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("HAM weights were not found; set --ham-checkpoint to a local file")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = _tensor_state(checkpoint)
    del checkpoint
    model = HAMBackbone(state, image_size)
    state["visual.positional_embedding"] = _resize_position_embedding(
        state["visual.positional_embedding"], model.visual.grid,
    )
    model.load_state_dict(state, strict=True)
    return model.float().eval()
