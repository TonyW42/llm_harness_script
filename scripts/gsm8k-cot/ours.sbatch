#!/bin/bash

#SBATCH --job-name=gsm8k_cot_ours             # Job name
#SBATCH --partition=GPU-shared           # Use A100 GPU partition (change if needed)
#SBATCH --gpus=h100-80:1                 # Request 4 GPUs per node
#SBATCH --time=08:00:00                  # Time limit (2 hours)
#SBATCH --output=output/gsm8k_cot_ours.out        # Standard output log
#SBATCH --error=error/gsm8k_cot_ours.err         # Error log

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
    --tasks gsm8k_cot \
    --device cuda:0 \
    --log_samples \
    --output_path output \
    --gen_kwargs '{"until": ["Q:", "</s>", "<|im_end|>", "<|end_of_text|>", "\n = = ="]}' \
    --batch_size 1 > output/gsm8k_cot_ours.out 
    # --add_pause_in_prompt \

