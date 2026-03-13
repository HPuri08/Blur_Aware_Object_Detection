#!/usr/bin/env bash

PYTHON="/home/hlcv_team038/miniconda3/envs/detr/bin/python"
PROJECT_DIR="/home/hlcv_team038"

$PYTHON $PROJECT_DIR/detr/main.py \
  --batch_size 2 \
  --no_aux_loss \
  --eval \
  --num_queries 50 \
  --resume $PROJECT_DIR/outputs/2_1_detr_kitti_clean_6/checkpoint.pth \
  --backbone resnet18 \
  --coco_path $PROJECT_DIR/detr_dataset/clean
