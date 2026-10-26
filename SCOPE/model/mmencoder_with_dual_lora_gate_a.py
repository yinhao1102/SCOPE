"""Backward-compatible imports for checkpoints and legacy experiment scripts."""

from .gd_lora import (
    GDLoRALinear as LoRALinear,
    GDLoRAMultiheadAttention as Dual_LoRAMultiheadAttention_gate,
    GDLoRAParameter as Dual_LoRAParameter,
    GDLoRATransformer as MMTransformer_with_duallora_gate,
    TextGDLoRATransformer as text_Transformer_with_duallora_gate,
)

__all__ = [
    "Dual_LoRAParameter",
    "Dual_LoRAMultiheadAttention_gate",
    "LoRALinear",
    "MMTransformer_with_duallora_gate",
    "text_Transformer_with_duallora_gate",
]
