#!/bin/bash
#SBATCH --job-name=bert-fine-tune
#SBATCH --output=logs/bert-fine-tune-%J.log
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1 #--gpus=1
#SBATCH --partition=gpu                   # partition to run on nodes that contain gpus


source ~/miniconda3/etc/profile.d/conda.sh 
conda activate pytorch_env
srun --nodes=1 --exclusive --gres=gpu:1 --ntasks=1 python run_fine_tune.py