#!/usr/bin/env bash

PYTHON="/home/hlcv_team038/miniconda3/envs/detr/bin/python"
PROJECT_DIR="/home/hlcv_team038"

$PYTHON $PROJECT_DIR/pretrain_resnet18.py \
    --data_dir $PROJECT_DIR/detr_dataset/clean/images/train \
    --epochs 100 \
    --batch_size 128 \
    --save_path $PROJECT_DIR/resnet/pretrained_simclr_r18_kitti_clean.pth
