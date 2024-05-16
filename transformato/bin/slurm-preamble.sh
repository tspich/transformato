#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu

source ~/.bashrc
conda activate fepAmber_ts
