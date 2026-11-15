"""Attach the paper's cardinality- and modality-specific fusion heads."""
import torch
from torch import nn

HEADS = ("mm_two_adapter", "mm_three_adapter", "mm_four_adapter")
MODALITIES = ("nir", "cp", "sk", "text")


def attach_msf(model, active_modalities=(0, 1, 2, 3), seed=2):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if any(hasattr(model, name) for name in HEADS):
        raise ValueError("Load an encoder-only model before attaching MSF")
    device = next(model.parameters()).device
    dimension = model.embed_dim
    # Keep all heads and their construction order for checkpoint compatibility.
    for name in HEADS:
        heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension, dimension // 4), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(dimension // 4, dimension // 16), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(dimension // 16, 1),
            )
            for _ in range(4)
        ])
        model._init_mm_adapter(heads)
        setattr(model, name, heads.to(device))
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    for size in range(2, len(active_modalities) + 1):
        for index in active_modalities:
            getattr(model, HEADS[size - 2])[index].requires_grad_(True)
    fusion_mode(model, True)


def fusion_mode(model, training):
    """Freeze encoder behavior as well as gradients during fusion training."""
    model.eval()
    for name in HEADS:
        if hasattr(model, name):
            getattr(model, name).train(training)


def heads_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if any(name.startswith(prefix + ".") for prefix in HEADS)
    }


def load_heads(model, checkpoint):
    state = checkpoint.get("heads", checkpoint.get("model", checkpoint))
    expected = set(heads_state(model))
    if set(state) != expected:
        raise ValueError("Fusion checkpoint must contain exactly the MSF heads")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(name in expected for name in incompatible.missing_keys):
        raise ValueError("Fusion checkpoint is incompatible with this model")
