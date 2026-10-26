#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${1:-ORBench}"
DATA_ROOT="${DATA_ROOT:-./data}"
PRETRAIN_CHOICE="${PRETRAIN_CHOICE:-ViT-B/16}"

python train.py \
  --name scope \
  --dataset_name "${DATASET_NAME}" \
  --root_dir "${DATA_ROOT}" \
  --pretrain_choice "${PRETRAIN_CHOICE}" \
  --loss_names mm_itc+mca \
  --lora_mode mlp \
  --lora_layers 12 \
  --lora_r1 8 \
  --lora_r2 8 \
  --num_loras 4 \
  --mca_loss_weight 0.1 \
  --mca_temperature 0.5
