#!/bin/bash

# Path to the `.py` file you want to run (including filename)
PYTHON_SCRIPT_PATH="/home/hlcv_team038/HLCV_Project/generate_blurred.py"

# Path to the Python binary of your conda environment
CONDA_PYTHON_BINARY_PATH="/home/hlcv_team038/miniconda3/envs/hlcv/bin/python"

# Run the Python script with the conda Python binary and pass arguments if needed
$CONDA_PYTHON_BINARY_PATH $PYTHON_SCRIPT_PATH --input_dir /home/hlcv_team038/data_object_image_2/training/image_2 --output_dir /home/hlcv_team038/generated_blurred_dataset

