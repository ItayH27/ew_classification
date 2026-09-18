#!/bin/bash
#SBATCH --job-name=sanity_check_severity
#SBATCH --output=sanity_severity.out
#SBATCH --error=sanity_severity.err
#SBATCH --partition=studentkillable
#SBATCH --time=30                     # minutes; CPU-only, no dataset download needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --gres=gpu:1

python3 -u ../sanity_check_severity.py

echo "Done."