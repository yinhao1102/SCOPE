#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-6}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
  --name gate_dual_lora_no_mm_kl_entropy \
  --output_dir ./logs \
  --root_dir /home/yinhao/data \
  --dataset_name ORBench \
  --loss_names mm_itc \
  --lora_mode mlp \
  --lora_layers 12 \
  --lora_r1 8 \
  --lora_r2 8 \
  --num_loras 4 \
  --mmt_depth 1 \
  --batch_size 64 \
  --num_epoch 60 \
  --milestones 20 50 \
  --gamma 0.1 \
  --monitor_four_entropy \
  "$@"
