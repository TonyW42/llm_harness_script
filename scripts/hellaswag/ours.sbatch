#!/bin/bash

#SBATCH --job-name=hellaswag_ours             # Job name
#SBATCH --partition=GPU-shared           # Use A100 GPU partition (change if needed)
#SBATCH --gpus=h100-80:1                 # Request 4 GPUs per node
#SBATCH --time=08:00:00                  # Time limit (2 hours)
#SBATCH --output=output/hellaswag_ours.out        # Standard output log
#SBATCH --error=error/hellaswag_ours.err         # Error log

# Load required modules
module load anaconda3

# Set conda environment directory and activate your environment
# export CONDA_PKGS_DIRS=/ocean/projects/cis240092p/$USER/conda_env
conda activate /ocean/projects/cis240092p/yw109/conda_env/pause_token

# Load CUDA module
module load cuda

# Avoid memory fragmentation
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python script with specified parameters
lm_eval --model hf \
    --model_args pretrained="/ocean/projects/cis240092p/yw109/pause_token/models/llama3_8b/ours" \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 1 > output/hellaswag_ours.out 

