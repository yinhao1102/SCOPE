#!/bin/bash
DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--name unireid_lora\
--dataset_name $DATASET_NAME \
--loss_names itc
