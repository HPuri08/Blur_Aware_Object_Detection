#!/usr/bin/env bash

PYTHON="/home/hlcv_team038/miniconda3/envs/detr/bin/python"
PROJECT_DIR="/home/hlcv_team038"

# Train DETR
$PYTHON $PROJECT_DIR/detr/main.py \
  --coco_path $PROJECT_DIR/detr_dataset/clear \
  --output_dir $PROJECT_DIR/outputs/2_1_detr_kitti_clean_7 \
  --backbone resnet18 \
  --batch_size 4 \
  --num_queries 50 \
  --epochs 40 \
  --lr 5e-5 \
  --lr_backbone 5e-6

