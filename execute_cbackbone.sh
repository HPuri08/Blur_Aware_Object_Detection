PYTHON="/home/hlcv_team038/miniconda3/envs/detr/bin/python"
PROJECT_DIR="/home/hlcv_team038"

$PYTHON $PROJECT_DIR/detr_dif_backbone/main.py \
    --coco_path $PROJECT_DIR/detr_dataset/clear \
    --epochs 20 \
    --output_dir $PROJECT_DIR/outputs/2_3_detr_kitti_clean_resnet_clean_1 \
    --backbone_weights $PROJECT_DIR/pretrained_models/resnet18_simclr_kitti_clean.pth \
    --batch_size 4 
