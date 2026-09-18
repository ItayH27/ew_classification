#!/bin/bash
#SBATCH --job-name=seed_sensitivity_segmentation
#SBATCH --output=seed_sensitivity_segmentation.out
#SBATCH --error=seed_sensitivity_segmentation.err
#SBATCH --partition=studentkillable
#SBATCH --time=360                    # minutes, adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gres=gpu:geforce_rtx_2080:1

source /home/yandex/DLWorkShop2026b/itayhazan/anaconda3/etc/profile.d/conda.sh
conda activate ew_DL

python3 -u ../seed_sensitivity_segmentation.py --epochs 15

echo "Done."
