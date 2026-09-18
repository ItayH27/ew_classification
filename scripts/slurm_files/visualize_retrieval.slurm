#!/bin/bash
#SBATCH --job-name=visualize_retrieval
#SBATCH --output=visualize_retrieval.out
#SBATCH --error=visualize_retrieval.err
#SBATCH --partition=studentkillable
#SBATCH --time=30                     # minutes; GEE fetch + single-image segmentation inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --gres=gpu:1

python3 -u ../visualize_retrieval.py

echo "Done."