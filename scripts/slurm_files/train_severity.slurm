#!/bin/bash
#SBATCH --job-name=train_severity
#SBATCH --output=train_severity.out
#SBATCH --error=train_severity.err
#SBATCH --partition=studentkillable
#SBATCH --time=30                     # minutes, adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000

# No --negative_dir flag exists here (unlike train_segmentation.py) — the negative
# samples already flow in automatically via the severity=0 rows in
# data/severity_labels.csv (see scripts/label_severity_samples.py --auto_label_negatives),
# joined against list_all_train_image_paths() inside src/severity/train.py.
python3 -u ../train_severity.py 

echo "Done."