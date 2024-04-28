#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 #--gpus=1
#SBATCH --output=logs/test_run-%J.out
#SBATCH --error=logs/test_run-%J.err
#SBATCH --job-name="Testing run script"

srun python prompt_tuning_script.py