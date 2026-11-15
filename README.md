# SCOPE

Model components for **SCOPE: A Contribution-aware and Efficient Framework for Multimodal Person Retrieval**.

- `SCOPE/model/gd_lora.py`: free-gate Dual-LoRA for visual and text transformers.
- `SCOPE/model/scope.py`: modality-specific visual tokenizers and shared backbone integration.
- `SCOPE/model/msf.py`: modality- and cardinality-specific fusion heads.
- `SCOPE/model/mca.py`: detached CE-gradient contribution targets and SDM + MCA objective.
- `SCOPE/model/objectives.py`: similarity distribution matching.

Requires Python 3.10+ and PyTorch 2.5+. Supply an already-loaded [HAM](https://github.com/MPI-Lab/HAM) / [CLIP](https://github.com/openai/CLIP) ViT backbone. Its positional embeddings must already match the input resolution. Images must be preprocessed and text must be supplied as CLIP token IDs; the model does not download or load external files.

```python
from SCOPE import SCOPE
from SCOPE.model import attach_msf, fusion_mode

# backbone: an already-loaded HAM/CLIP-compatible ViT module.
model = SCOPE(backbone, rank_magnitude=8, rank_direction=8)
features = model({"rgb": rgb_images, "sk": sketch_images, "text": text_token_ids})

# Call after adapting/loading the encoder. This freezes the encoder and
# initializes the fusion heads; learned fusion requires trained head weights.
attach_msf(model, active_modalities=(2, 3))  # NIR=0, CP=1, SK=2, TEXT=3
fusion_mode(model, training=False)
fused, logits, weights = model.msf_fuse_by_indices(
    [None, None, features["sk"], features["text"]], (2, 3)
)
```

The default encoder uses rank 8 + 8, AB-norm rank reweighting, learned positive gate strength, and output-channel RMS calibration. It has no additive direction residual. Backbone transformer weights remain frozen; low-rank modules, visual tokenizers, layer normalization, and projection parameters provide adaptation.

MSF combines raw features before final cosine normalization. For contribution alignment, `gallery_context` accepts detached RGB reference features and identity labels from training data. `gradient_contribution` estimates utility derivatives at uniform fusion weights. The caller supplies training-calibrated temperatures to `fusion_loss`; its default MCA coefficient is 1.5. During fusion-only optimization, encode inputs without gradients and use `fusion_mode(model, training=True)` to keep the encoder in evaluation mode.

This repository contains model source only.
