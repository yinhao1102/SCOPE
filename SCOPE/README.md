
## Overview

SCOPE is a unified person retrieval framework for heterogeneous and compositional queries. One model supports arbitrary modality combinations on ORBench and RGB-Sketch-Text retrieval on Tri-CUHK-PEDES and Tri-RSTPReid.

The framework contains three main components:

- **GD-LoRA**: gated dual low-rank adaptation for modality-specific representation learning.
- **MSF**: modality-sensitive fusion that estimates sample-wise modality contributions.
- **MCA loss**: modality contribution alignment based on leave-one-modality-out retrieval degradation.

## Supported Datasets

| Dataset | Gallery | Query modalities | Training and evaluation |
|---|---|---|---|
| ORBench | RGB | NIR, CP, Sketch, Text and their combinations | Yes |
| Tri-CUHK-PEDES | RGB | Sketch, Text, Sketch+Text | Yes |
| Tri-RSTPReid | RGB | Sketch, Text, Sketch+Text | Yes |

## Installation

```bash
git clone <repository-url>
cd SCOPE

conda create -n scope python=3.9 -y
conda activate scope
pip install -r requirements.txt
```

The current code has been tested with Python 3.9, PyTorch 2.5.1, and torchvision 0.20.1.

## Data Preparation

Please follow each dataset's license and access requirements. Dataset files are not redistributed by this repository.

### ORBench

Download ORBench from the official ReID5o/ORBench repository.

### Tri-CUHK-PEDES

1. Obtain the RGB images and `reid_raw.json` by following the CUHK-PEDES preparation page.
2. Download the sketch modality released by UNIReID.
3. Place the sketches in `CUHK-PEDES/imgs-sketch2/` while preserving the RGB image subdirectory structure.

### Tri-RSTPReid

1. Download the base dataset from the official RSTPReid repository.
2. Download the sketch modality from the UNIReID release.
3. Place the sketches in `RSTPReid/img-sketch2/` while preserving the RGB image subdirectory structure.

The final directory layout must be:

```text
data/
├── ORBench/
│   ├── vis/
│   ├── nir/
│   ├── cp/
│   ├── sk/
│   ├── train_annos.json
│   └── test_gallery_and_queries.json
├── CUHK-PEDES/
│   ├── imgs/
│   ├── imgs-sketch2/
│   └── reid_raw.json
└── RSTPReid/
    ├── imgs/
    ├── img-sketch2/
    └── data_captions.json
```

Set `DATA_ROOT` when the datasets are stored elsewhere:

```bash
export DATA_ROOT=/path/to/data
```

## Pretrained Models

### Recommended: HAM ReID Pretraining

We recommend initializing SCOPE with the ReID pretrained model released by MPI-Lab/HAM, which trains a CLIP ViT-B/16 retrieval model.

After downloading the checkpoint, provide its local path through `PRETRAIN_CHOICE`:

```bash
export PRETRAIN_CHOICE=/path/to/ham_reid_pretrain.pth
```

## Training

`train.sh` is the unified one-command launcher. It uses `./data` and OpenAI CLIP by default, or the `DATA_ROOT` and `PRETRAIN_CHOICE` environment variables shown above.

```bash
# ORBench
bash train.sh ORBench

# Tri-CUHK-PEDES
bash train.sh Tri-CUHK-PEDES

# Tri-RSTPReid
bash train.sh Tri-RSTPReid
```

Recommended HAM initialization:

```bash
DATA_ROOT=/path/to/data \
PRETRAIN_CHOICE=/path/to/ham_reid_pretrain.pth \
bash train.sh ORBench
```

Training outputs are saved to:

```text
outputs/<dataset>/<timestamp>_scope/
├── configs.yaml
├── R1_best.pth
└── mAP_best.pth
```

The default schedule first optimizes GD-LoRA and modality tokenizers. MSF and MCA optimization starts at epoch 25. Use `--fusion_start_epoch` to change this boundary.

## Evaluation

Evaluate a run directory with one command:

```bash
bash test.sh outputs/ORBench/<run-directory>
bash test.sh outputs/Tri-CUHK-PEDES/<run-directory>
bash test.sh outputs/Tri-RSTPReid/<run-directory>
```

When the dataset is outside `./data`, override the path without editing the saved config:

```bash
DATA_ROOT=/path/to/data bash test.sh pretrained/ORBench
```

Evaluate `mAP_best.pth` instead of the default `R1_best.pth`:

```bash
bash test.sh outputs/ORBench/<run-directory> mAP_best.pth
```

For a downloaded checkpoint:

```bash
bash test.sh pretrained/ORBench
```

Use average fusion instead of MSF for ablation:

```bash
python test.py \
  --config_file pretrained/ORBench/configs.yaml \
  --checkpoint R1_best.pth \
  --no_msf
```

ORBench reports all one-, two-, three-, and four-modality query combinations. The Tri datasets report Text-to-RGB, Sketch-to-RGB, and Text+Sketch-to-RGB retrieval.

## Repository Structure

```text
SCOPE/
├── model/
│   ├── gd_lora.py                 # GD-LoRA
│   └── build_unireid_lora.py      # SCOPE, MSF, and MCA
├── reid5o_datasets/               # ORBench pipeline
├── tri_datasets/                  # Tri-CUHK-PEDES and Tri-RSTPReid
├── utils/
│   ├── reid5o_metrics.py          # ORBench metrics
│   └── tri_metrics.py             # Tri-dataset metrics
├── train.py
├── test.py
├── train.sh
└── test.sh
```
