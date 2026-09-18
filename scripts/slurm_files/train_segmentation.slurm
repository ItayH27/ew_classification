#!/bin/bash
#SBATCH --job-name=train_segmentation
#SBATCH --output=train_segmentation.out
#SBATCH --error=train_segmentation.err
#SBATCH --partition=studentkillable
#SBATCH --time=240                    # minutes, adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gres=gpu:geforce_rtx_2080:1

source /home/yandex/DLWorkShop2026b/itayhazan/anaconda3/etc/profile.d/conda.sh
conda activate ew_DL

python3 -u ../train_segmentation.py --alpha 0.3 --pos_weight 14 --seed 42 \
    --batch_size 2 --accumulation_steps 4 \
    --negative_dir /home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/dataset/negative \
    --negative_fraction 0.2

echo "Done."