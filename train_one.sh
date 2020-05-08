#!/bin/bash
#SBATCH --time=01:05:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=5000
#SBATCH --partition=gpu
#SBATCH -o output/logs/runtime_%j.out
#SBATCH -e output/logs/runtime_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maldives003@gmail.com

module load cudnn/cuda-8.0/6.0
module load anaconda3/4.4.0

source activate selene-gpu

python -u run_best.py
