# Environmental Warfare Classifier

A deep-learning tool that verifies and quantifies environmental damage (agricultural/forest) from
conflicts, using satellite imagery and public media/conflict records. By Eran Shufaru and Itay Hazan.
Click here for a [`Full Demo`](https://itaytau27-ew-classifier.hf.space).

## Pipeline overview

Given a location, event date (or a local satellite scene), the pipeline runs four stages end to end:

1. **Reference retrieval** (`src/retrieval/`) — fetches pre-event and post-event Sentinel-2 imagery
   (via Google Earth Engine) for a 10 km AOI around the given lat/lon, cloud-masked and mosaicked over a
   pre-event window (`event_date - 30d` .. `event_date - 1d`) and a post-event window (`event_date` ..
   `event_date + 21d`).
2. **Semantic segmentation** (`src/segmentation/`) — a fine-tuned `UNet2DModel` (`BurnScarSegmenter`)
   identifies burned/damaged vegetation pixels in the post-event image.
3. **Severity classification** (`src/severity/`) — a `StandardScaler` + multinomial logistic regression
   over mean NDVI/NBR (whole-image and masked-region) plus burned-area fraction scores overall damage
   0 (no damage) to 3 (maximal).
4. **Attribution** (`src/attribution/`) — searches the World News API for articles near the
   coordinates/date, dedupes near-identical wire-service reprints (TF-IDF cosine clustering), then asks
   Gemini per-article whether it describes an environmental warfare incident there, aggregating verdicts
   into a percentage score.

`src/pipeline.py`'s `run_pipeline(...)` composes all four stages. It's exposed via a CLI
(`scripts/run_pipeline.py`) and a local FastAPI web app (`web/`).

**Cluster note:** development targets the TAU Slurm cluster with limited memory/GPU — the
`studentkillable` partition mixes old Maxwell GPUs (which silently fall back to CPU under the current
driver) with working RTX 2080 Ti cards, so `.slurm` files pin `--gres=gpu:geforce_rtx_2080:1` explicitly
rather than a bare `--gres=gpu:1`.

## Setup

Development happens in the `ew_DL` conda environment, not the base environment or system `python3`:

```bash
conda activate ew_DL
pip install -r requirements.txt
cp env.example .env   # then fill in the values below
```

Fill in `.env`:

| Variable | Needed for | Notes |
|---|---|---|
| `GEE_PROJECT` | Stage 1 (retrieval) | Google Cloud project ID |
| `GEE_SERVICE_ACCOUNT` + `GEE_KEY_FILE` | Stage 1, cluster/CI | Service-account auth (recommended off-laptop) |
| *(none — run `earthengine authenticate` once)* | Stage 1, local dev | Interactive auth: set only `GEE_PROJECT` |
| `HF_TOKEN` | Segmentation training | Optional; avoids HF Hub rate limits |
| `WORLD_NEWS_API_KEY` | Stage 4 (attribution) | https://worldnewsapi.com |
| `GEMINI_API_KEY` | Stage 4 (attribution) | https://aistudio.google.com/apikey |

Stage 4's keys are only required for GEE-mode pipeline runs — local-mode runs (`--local_tif`) skip
attribution entirely (no lat/lon/event_date to search on) and don't need either key.

There is no test suite, linter, or formatter configured in this repo — use the `sanity_check_*.py`
scripts below to validate each stage.

Run everything from the repo root — scripts insert the repo root onto `sys.path` themselves, so no
`PYTHONPATH` setup is needed.

## Sanity checks

```bash
python scripts/visualize_retrieval.py        # RGB/NDVI/NBR for a known damaged location — sanity-checks GEE auth
python scripts/sanity_check_segmentation.py   # dataset, model, loss, metrics
python scripts/sanity_check_severity.py       # dataset, model roundtrip, training, checkpoint reload
python scripts/sanity_check_attribution.py    # World News search, Gemini classify, dedup, retry logic
```

## Training

### 1. Segmentation model

Fine-tunes `BurnScarSegmenter` (a repurposed `diffusers.UNet2DModel`) on the
`ibm-nasa-geospatial/hls_burn_scars` dataset (downloaded from HF Hub on first run), optionally mixed with
GEE-sourced negative (no-damage) samples.

```bash
# Bulk-sample negatives first (optional but used by the production config below)
python scripts/fetch_negative_samples.py --n_target 150 --out_dir dataset/negative

# Default run (30 epochs, batch 8, lr 1e-4, alpha 0.5, pos_weight 10)
python scripts/train_segmentation.py

# Production config (see hyperparameter sweep below), reproducible run
python scripts/train_segmentation.py --alpha 0.3 --pos_weight 14 --seed 42 \
    --batch_size 2 --accumulation_steps 4 \
    --negative_dir dataset/negative --negative_fraction 0.2
```

Or submit the production config to the cluster: `sbatch scripts/slurm_files/train_segmentation.slurm`.

Key hyperparameters (`scripts/train_segmentation.py --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 30 | training epochs |
| `--batch_size` | 8 | samples/step (production uses 2, small-GPU) |
| `--accumulation_steps` | 1 | effective batch = batch_size × steps (production: 4 → effective batch 8) |
| `--lr` | 1e-4 | AdamW learning rate |
| `--alpha` | 0.5 | `CombinedLoss` weight: BCE vs. Dice |
| `--pos_weight` | 10.0 | BCE positive-class weight (class imbalance) |
| `--negative_dir` / `--negative_fraction` | off / 0.0 | mix in GEE negatives via `WeightedRandomSampler` |
| `--seed` | unseeded | set for reproducible runs |

Best checkpoint (by validation IoU) is saved to `<output_dir>/best_model/` (diffusers format) plus
`training_state.pt` (optimizer state + epoch, for resuming); default `output_dir` is
`checkpoints/burn_scar/`.

**Hyperparameter sweep and seed sensitivity:** `scripts/sweep_segmentation_hparams.py` cheaply ranks
alpha/pos_weight candidates (15 epochs each, fixed seed 42, production negative-mixing/batch config) —
`B_a0.30_pw14` won (val IoU 0.714 vs. 0.706 for the alpha=0.3/pos_weight=7 baseline; see
`seg_hyp_param_summery.csv`), which is why it's the production config above.
`scripts/seed_sensitivity_segmentation.py` then holds that config fixed and re-runs across 5 seeds
(`7, 42, 67, 69, 420`) to measure how much of the metric spread is seed noise vs. real signal
(val IoU ranged 0.691–0.708 across seeds; see `seg_seed_summary.csv`) — both scripts write one row per
run to `summary.csv` immediately, so a run surviving preemption on `studentkillable` doesn't lose prior
results.

```bash
python scripts/sweep_segmentation_hparams.py
python scripts/seed_sensitivity_segmentation.py --epochs 30 --seeds 1 2 3
```

### 2. Severity classifier

Trained on manually-labeled HLS burn-scar samples plus optional GEE-sourced negatives.

```bash
# 1. Manually label severity (0=no damage .. 3=maximal) for a deterministic seeded sample
#    (seed 42, 200 burn-positive images), resumable — appends to data/severity_labels.csv
python scripts/label_severity_samples.py
python scripts/label_severity_samples.py --n_negative_samples 50 --auto_label_negatives  # + GEE negatives

# 2. Train (stratified 5-fold CV, then refit on all labels)
python scripts/train_severity.py
```

Or submit to the cluster: `sbatch scripts/slurm_files/train_severity.slurm`.

Key hyperparameters: `--n_splits` (default 5, auto-shrinks if a class has fewer members) and `--seed`
(default 42, for CV shuffling). Saves a joblib bundle to `checkpoints/severity/model.joblib` (default
`--output_dir`).

### 3. Segmentation model comparison (vs. Prithvi)

Compares `BurnScarSegmenter` against IBM/NASA's pretrained `Prithvi-EO-2.0-300M-BurnScars` on the HLS
Burn Scar validation split. Needs its own conda env (`terratorch` pulls in a heavy dependency chain kept
out of `ew_DL`):

```bash
conda create -n compare_seg python=3.11 -y
conda activate compare_seg
pip install -r compare_seg/requirements.txt

python compare_seg/compare.py --models ours --limit 5   # quick smoke test, no Prithvi download needed
python compare_seg/compare.py                           # full comparison, both models
```

Or submit to the cluster: `sbatch compare_seg/compare.slurm` (run from `compare_seg/`).

Reports both a **full** (264-chip) and a **clean** (~87-chip, leak-free) IoU/Dice/FPR/FNR, since 67% of
this repo's "validation" chips are actually in Prithvi's own training split — read the clean number when
judging generalization. Bottom line from the last run: our 56.6M-param model (clean IoU 0.803) is
essentially on par with the 324M-param Prithvi model (clean IoU 0.813). Full detail in
[`compare_seg/README.md`](compare_seg/README.md).

## Running the full pipeline

### CLI

```bash
# GEE mode: fetch pre/post Sentinel-2 imagery for a lat/lon + event date
python scripts/run_pipeline.py --lat 48.9001 --lon 37.7249 --event_date 2024-07-19

# Local mode: run directly on a local HLS *_merged.tif, bypassing GEE (and skipping attribution)
python scripts/run_pipeline.py --local_tif dataset/validation/<id>_merged.tif

# Machine-readable-only output, safe to pipe
python scripts/run_pipeline.py --local_tif <tif> --output json | python -m json.tool
```

`--output {json,png,both}` (default `both`) controls what's produced besides the always-printed
human-readable summary; `--output_dir` controls where the PNG grid is saved; `--mask_threshold` (default
0.5) sets the segmentation sigmoid threshold.

### Web UI

```bash
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Serves `web/static/index.html`, backed by `POST /api/run` (GEE mode) and `POST /api/run_local` (upload a
`*_merged.tif`). The `-m` form matters — it's what puts the repo root on `sys.path`. Single worker only:
the segmentation/severity models are cached as module-level singletons, so extra workers would just
duplicate model memory for no throughput benefit at this scale.
