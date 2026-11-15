# SCOPE

Model implementation of **SCOPE: A Contribution-aware and Efficient Framework for Multimodal Person Retrieval**, using the **HAM-pretrained ViT-B/16** backbone.

## HAM pretrained weights

Download the **ReID pretrained model** linked in the [official HAM README](https://github.com/MPI-Lab/HAM/blob/main/readme.md) and place the checkpoint at `pretrained/ham.pth`, or pass its local location with `--ham-checkpoint`. The loader accepts the original HAM checkpoint and a backbone tensor state dictionary. Weights are not included in this repository.

## Quick start

Requires Python 3.10+ and PyTorch 2.5+.

```bash
python -m pip install -r requirements.txt
bash run.sh --ham-checkpoint pretrained/ham.pth
```

The command loads HAM, builds SCOPE with rank 8 + 8, and runs a synthetic forward check for RGB, NIR, CP, SK, Text, and all 11 multimodal fusion subsets. It uses an available CUDA device or falls back to CPU. Successful execution prints `"status": "passed"` and feature shapes. This checks model execution; it does not train the model or report retrieval performance. MSF heads are newly initialized unless trained weights are supplied.

```bash
# Run on CPU explicitly.
bash run.sh --ham-checkpoint pretrained/ham.pth --device cpu

# Optionally check a matched pair of trained SCOPE encoder and MSF checkpoints.
bash run.sh --ham-checkpoint pretrained/ham.pth \
  --encoder-checkpoint checkpoints/encoder.pth \
  --fusion-checkpoint checkpoints/msf.pth
```

## Use the model

```python
from SCOPE import SCOPE
from SCOPE.model import attach_msf, fusion_mode

model = SCOPE.from_pretrained("pretrained/ham.pth")
# Images are preprocessed tensors; text is supplied as token IDs.
features = model({"rgb": rgb_images, "sk": sketch_images, "text": text_token_ids})

# Initialize MSF after adapting/loading the encoder; this freezes the encoder.
attach_msf(model, active_modalities=(2, 3))  # NIR=0, CP=1, SK=2, TEXT=3
fusion_mode(model, training=False)
fused, logits, weights = model.msf_fuse_by_indices(
    [None, None, features["sk"], features["text"]], (2, 3)
)
```

Default image size is 384 × 128. The free gate uses AB-norm rank reweighting, learned positive strength, and output-channel RMS calibration, with no additive direction residual. MSF combines raw features before final cosine normalization.

For fusion training, `fusion_loss` implements SDM + MCA with a default MCA coefficient of 1.5. Its detached contribution targets use uniform reference fusion and training RGB reference features. Supply training-calibrated temperatures, encode inputs without gradients, and call `fusion_mode(model, training=True)` to keep the encoder in evaluation mode.

## Model files

- `SCOPE/model/ham.py`: local HAM weight loading and backbone construction.
- `SCOPE/model/scope.py`: SCOPE encoders and backbone integration.
- `SCOPE/model/gd_lora.py`: free-gate Dual-LoRA.
- `SCOPE/model/msf.py`: modality-specific fusion heads.
- `SCOPE/model/mca.py`: modality contribution alignment.
- `SCOPE/model/objectives.py`: similarity distribution matching.

## Acknowledgements

We thank the authors of [IRRA](https://github.com/anosorae/IRRA), [HAM](https://github.com/MPI-Lab/HAM), and [CLIP](https://github.com/openai/CLIP) for making their research and code available. SCOPE uses the HAM-pretrained backbone, builds on the CLIP architecture, and adopts the SDM objective introduced by IRRA. Notices for adapted code are included in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
