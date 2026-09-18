# compare_seg

Compares our from-scratch `BurnScarSegmenter` (`src/segmentation/`) against IBM/NASA's pretrained
[`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars)
on the HLS Burn Scar Scenes validation split.

## The leakage caveat

Prithvi-BurnScars was fine-tuned on IBM's own train/val/test partition of the same 804-chip
dataset pool our `dataset/training/`+`dataset/validation/` folders use — but it's a *different*
partition. Diffing chip IDs shows **177 of our 264 `dataset/validation/` chips (67%) are in IBM's
own training split**, so Prithvi has already seen most of what this repo calls "validation".

To keep the comparison fair, `compare.py` reports two numbers per model:
- **full** — all 264 chips in `dataset/validation/` (what "the validation split" means in this repo).
- **clean** — the ~87-chip subset confirmed absent from IBM's `splits/train.txt` (i.e. genuinely
  unseen by Prithvi during its own training). This is the more trustworthy number for comparing the
  two models' real generalization performance.

Both are computed from the same inference pass (no extra compute), and both are written to
`results/summary.json` / `results/per_chip.csv`.

## One-time setup

This comparison needs [`terratorch`](https://github.com/IBM/terratorch) to load Prithvi, which pulls
in a heavy dependency chain (pytorch-lightning, torchgeo, albumentations, geopandas, etc.) that we
don't want in the shared `ew_DL` env used by the rest of the pipeline. It gets its own dedicated env:

```bash
source /home/yandex/DLWorkShop2026b/itayhazan/anaconda3/etc/profile.d/conda.sh
conda create -n compare_seg python=3.11 -y
conda activate compare_seg
pip install -r compare_seg/requirements.txt
```

## Usage

```bash
conda activate compare_seg

# Quick smoke test (no download of the 1.3GB Prithvi checkpoint needed)
python compare.py --models ours --limit 5

# Full comparison, both models (downloads the Prithvi checkpoint on first run)
python compare.py

# Options
python compare.py --help
```

Or submit to the cluster (after the one-time env setup above):

```bash
cd compare_seg
sbatch compare.slurm
```

## Output

- Console: a summary table (iou/dice/fpr/fnr/latency) for each model x {full, clean}, plus a
  separate, distinctly-labeled block per model for the negative-sample (no-damage specificity) pass.
- `results/summary.json` — aggregate metrics for both eval sets and both models (`"full"`/`"clean"`),
  plus a `"negatives"` sub-dict per model (`n_chips`/`fpr`/`avg_latency_s` only), plus run metadata.
- `results/per_chip.csv` — per-chip IoU/Dice/FPR/FNR metrics with an `in_clean_subset` column, for
  the HLS burn-scar comparison.
- `results/negative_per_chip.csv` — per-chip `fpr`/`latency_s` only, for the negative-sample
  comparison. Kept in a separate file (not appended to `per_chip.csv`) since IoU/Dice/FNR are
  degenerate on these all-negative targets and `in_clean_subset` has no meaning for them.

## Results

Full run on all 264 validation chips (`results/summary.json`, from a checkpoint trained on the
`neg_samples` branch — see the note below on how this compares to an earlier checkpoint):

| Metric | Ours (56.6M params) | Prithvi (324.2M params) |
|---|---|---|
| IoU — full (264 chips) | 0.765 | 0.811 |
| IoU — **clean (87 leak-free chips)** | **0.777** | **0.813** |
| Dice — clean | 0.850 | 0.884 |
| FPR — clean | 0.018 | 0.017 |
| FNR — clean | 0.126 | 0.069 |

**Read the "clean" row, not "full."** Prithvi was fine-tuned on IBM's own split of this same
dataset, and 177 of our 264 "validation" chips are in *its* training set — on those chips it's
partly recalling data it already fit, which is why the full-264 gap (0.765 vs 0.811) looks bigger
than it should. On the 87 chips confirmed unseen by Prithvi during its own training, the gap
narrows but doesn't close: **0.777 vs 0.813, a ~3.6-point IoU difference**, driven mainly by FNR
(missed burn scars: 0.126 vs 0.069) rather than false positives (FPR is essentially tied, 0.018 vs
0.017).

**Note on this checkpoint vs. an earlier run:** an earlier checkpoint (see git history) scored
0.803 clean IoU / 0.089 clean FNR — closer to Prithvi. The current checkpoint was retrained on the
`neg_samples` branch (mixing in GEE negative samples during training, plus other hyperparameter/seed
changes — see that branch's commit history), and while it's now much better on the specificity axis
below, it's given up some sensitivity on this axis (more missed burn scars) relative to the earlier
checkpoint. Worth treating as a real trade-off to investigate, not just noise — Prithvi's own numbers
are unchanged run-to-run (fixed checkpoint), so the shift here is entirely attributable to our
retraining.

**Bottom line:** on a fair (leak-free) comparison, our from-scratch 56.6M-param model trails Prithvi
by a real but still fairly modest IoU gap, at ~5.7x fewer parameters — driven by a higher miss rate,
not by hallucinating damage. It might still be worth investigating whether the negative-sample
mixing that improved specificity (see below) can be tuned to recover some of this lost sensitivity.

*(Note: our model was only ever trained at 256x256. `compare.py` downsamples its input to 256 before
inference and upsamples the prediction back to native 512x512 to share a pixel grid with the ground
truth and with Prithvi's native-512 output — running it directly at native 512 is out-of-distribution
and collapses its IoU to near-zero on several chips, which was caught and fixed during development.)*

## Negative-sample specificity

The comparison above only measures sensitivity (finding real burn scars) — it says nothing about
whether a model hallucinates damage on undamaged imagery. `compare.py` also runs both models against
the GEE-sourced no-damage chips in `dataset/negative/validation/` (all-zero ground-truth masks, see
`src/retrieval/negative_sampling.py`) and reports **false-positive rate only** — IoU/Dice/FNR are
degenerate (undefined-in-spirit, ≈1.0 or 0/0) on all-negative targets, so this is deliberately a
**separate table, not blended into the IoU comparison above**: the two evals answer different
questions, and Prithvi — fine-tuned only on burn-scar-positive chips, unlike our model, which sees
GEE negatives during training via `--negative_dir`/`--negative_fraction` — isn't being judged unfairly
by being tested on this axis too; a specificity gap here would be a real, meaningful finding.

These chips are native 256x256 (vs. HLS's 512x512). Our model evaluates at that resolution natively
(no resize needed — a bonus over the HLS case). Prithvi is upsampled to its own 512x512 fine-tuning
resolution for inference, with its prediction downsampled back to native 256 for comparison, so each
model is evaluated near its own trained resolution.

| Metric | Ours (56.6M params) | Prithvi (324.2M params) |
|---|---|---|
| n (negative chips) | 22 | 22 |
| FPR | **0.31%** | **22.30%** |
| Avg latency | 0.018s | 0.027s |

A large, non-marginal gap. Per-chip results (`results/negative_per_chip.csv`) show Prithvi's false
positives aren't spread evenly — they concentrate on farmland chips (Iowa cropland up to FPR 0.86;
Punjab cropland up to 0.78), while its forest/savanna chips (Kenya, Vietnam, France) mostly stay low
(0.002–0.07). Ours stays uniformly low everywhere, worst case 0.038. The likely mechanism: bare or
harvested farmland has low NDVI/NBR, spectrally similar to a burn scar, and Prithvi — fine-tuned only
on genuine burn-scar-positive/negative pixels within its own dataset — never had the chance to learn
that distinction, while our model was trained directly on GEE negative farmland chips.

**Caveat, mirroring the leakage caveat above but in the other direction:** this isn't a fully
apples-to-apples specificity test either. Our model has literally seen negative chips from these same
source regions (Iowa farmland, Punjab, etc. — see `NEGATIVE_SAMPLE_REGIONS` in
`src/retrieval/negative_sampling.py`) during training, while Prithvi never had negative examples at
all. So read this as "our model was trained for this task and Prithvi wasn't," not as evidence our
model generalizes better in general — the gap is expected given each model's training setup, and (per
the note in Results above) may be part of why our current checkpoint's burn-scar sensitivity (FNR)
regressed slightly relative to an earlier checkpoint trained without negative-sample mixing.
