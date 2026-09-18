#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --output=sanity.out
#SBATCH --error=sanity.err
#SBATCH --partition=studentkillable
#SBATCH --time=180                    # minutes, adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000                   # give it a healthy buffer for decompression etc.
#SBATCH --gres=gpu:1

python3 -u ../sanity_check_segmentation.py

echo "Done."