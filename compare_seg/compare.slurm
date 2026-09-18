#!/bin/bash
#SBATCH --job-name=compare_seg
#SBATCH --output=compare_seg.out
#SBATCH --error=compare_seg.err
#SBATCH --partition=studentkillable
#SBATCH --time=90                     # minutes; ~1.3GB Prithvi download + terratorch import overhead dominate
                                       # (also covers the small negative-sample specificity pass, ~22 extra chips)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gres=gpu:1

# One-time setup (run once before first sbatch, not part of this job — see compare_seg/README.md):
#   source /home/yandex/DLWorkShop2026b/itayhazan/anaconda3/etc/profile.d/conda.sh
#   conda create -n compare_seg python=3.11 -y
#   conda activate compare_seg
#   pip install -r /home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/compare_seg/requirements.txt

source /home/yandex/DLWorkShop2026b/itayhazan/anaconda3/etc/profile.d/conda.sh
conda activate compare_seg

python3 -u /home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/compare_seg/compare.py

echo "Done."
