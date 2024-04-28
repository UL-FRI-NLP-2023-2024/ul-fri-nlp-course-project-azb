#!/bin/bash
#SBATCH --job-name=jupyter-notebook
#SBATCH --output=jupyter-notebook-%J.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=4


# get tunneling info

port=8888
node=$(hostname -s)
user=$(whoami)

module load cuda

# run jupyter notebook
jupyter-notebook --no-browser --port=${port} --ip=${node}