#!/bin/bash
#SBATCH --job-name=sweep_segmentation
#SBATCH --output=sweep_segmentation.out
#SBATCH --error=sweep_segmentation.err
#SBATCH --partition=studentkillable
#SBATCH --time=360                    # minutes, adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gres=gpu:geforce_rtx_2080:1

source /home/yandex/DLWorkShop2026b/itayhazan/anaconda3/etc/profile.d/conda.sh
conda activate ew_DL

python3 -u ../sweep_segmentation_hparams.py --epochs 15

echo "Done."
