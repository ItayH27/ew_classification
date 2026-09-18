"""
CLI entry point for the full retrieval -> segmentation -> severity (-> attribution stub)
pipeline (src/pipeline.py).

Usage:
    # GEE mode: fetch pre/post Sentinel-2 imagery for a lat/lon + event date
    python scripts/run_pipeline.py --lat 48.9001 --lon 37.7249 --event_date 2024-07-19

    # Local mode: run directly on a local HLS *_merged.tif, bypassing GEE
    python scripts/run_pipeline.py --local_tif dataset/validation/<id>_merged.tif

    # Machine-readable output only (stdout carries only the JSON blob, safe to pipe)
    python scripts/run_pipeline.py --local_tif <tif> --output json | python -m json.tool

Outputs:
    A human-readable summary (severity, burn area, NDVI/NBR deltas, attribution status)
    printed for the human running the command, plus optionally a JSON blob on stdout
    (--output json/both) and/or a PNG visualization grid (--output png/both) saved to
    --output_dir.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from src.pipeline import render_human_readable, run_pipeline, summarize_result
from src.visualization import normalize_rgb, overlay_burn_mask

DEFAULT_OUTPUT_DIR = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/scripts/slurm_files"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full environmental-damage pipeline for a single location/date "
        "or a local HLS scene."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local_tif", type=str, help="Path to a local HLS *_merged.tif; bypasses GEE.")
    mode.add_argument("--lat", type=float, help="Latitude (requires --lon and --event_date).")
    p.add_argument("--lon", type=float, help="Longitude (required with --lat).")
    p.add_argument("--event_date", type=str, help="ISO-8601 event date YYYY-MM-DD (required with --lat).")
    p.add_argument("--output", choices=["json", "png", "both"], default="both",
                   help="What to produce besides the human-readable summary (default: both).")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Directory for the saved PNG (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--mask_threshold", type=float, default=0.5,
                   help="Sigmoid threshold for the segmentation mask (default: 0.5).")
    p.add_argument("--keywords", type=str, default=None,
                   help="Comma-separated World News API search terms for attribution, e.g. "
                   "'wildfire,fire,burning' (joined into the API's OR-query syntax). "
                   "Default: worldnews_client._DEFAULT_KEYWORDS.")
    p.add_argument("--attribution_mode", choices=["yes_no", "structured"], default="yes_no",
                   help="Attribution classification mode (default: yes_no). \"yes_no\" asks a "
                   "single yes/no per article; \"structured\" asks for a per-article JSON "
                   "verdict (conflict_activity, environmental_destruction, deliberate, "
                   "actor_claimed, evidence_quote, confidence) distinguishing deliberate "
                   "environmental destruction from incidental damage near fighting.")
    p.add_argument("--domain_prompt", type=str, default=None,
                   help="Path to a text file containing the domain-definition portion of the "
                   "Gemini classifier prompt for attribution, i.e. what counts as an incident. "
                   "Works in both modes. Default depends on --attribution_mode: "
                   "gemini_client._YES_NO_DOMAIN_PROMPT or gemini_client._STRUCTURED_DOMAIN_PROMPT.")
    p.add_argument("--output_contract", type=str, default=None,
                   help="Path to a text file containing the output-contract portion of the "
                   "Gemini classifier prompt, i.e. the JSON field descriptions. Only takes "
                   "effect with --attribution_mode structured — silently ignored in yes_no "
                   "mode, whose output contract is fixed (gemini_client._YES_NO_OUTPUT_CONTRACT). "
                   "Default: gemini_client._STRUCTURED_OUTPUT_CONTRACT.")
    args = p.parse_args()

    if args.lat is not None and (args.lon is None or args.event_date is None):
        p.error("--lat requires both --lon and --event_date.")

    if args.keywords is not None:
        terms = [term.strip() for term in args.keywords.split(",") if term.strip()]
        if not terms:
            p.error("--keywords must contain at least one non-empty, comma-separated term.")
        args.keywords = " OR ".join(terms)

    if args.domain_prompt is not None:
        try:
            with open(args.domain_prompt, "r") as f:
                args.domain_prompt = f.read()
        except OSError as e:
            p.error(f"--domain_prompt: could not read {args.domain_prompt!r}: {e}")

    if args.output_contract is not None:
        try:
            with open(args.output_contract, "r") as f:
                args.output_contract = f.read()
        except OSError as e:
            p.error(f"--output_contract: could not read {args.output_contract!r}: {e}")

    return args


def _slug(args: argparse.Namespace) -> str:
    if args.local_tif:
        return os.path.basename(args.local_tif).replace("_merged.tif", "")
    return f"{args.lat}_{args.lon}_{args.event_date}".replace(" ", "")


def _save_png(result: dict, out_path: str) -> None:
    index_cmap, index_norm = "RdYlGn", mcolors.Normalize(vmin=-1, vmax=1)

    if result["mode"] == "gee":
        # Predicted mask on the left spanning both rows; RGB/NDVI/NBR to its right, one
        # column per index, pre-event on top and post-event below (paired vertically).
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, width_ratios=[1.3, 1, 1, 1])

        ax_mask = fig.add_subplot(gs[:, 0])
        overlay = overlay_burn_mask(result["post"]["rgb"], result["mask"])
        ax_mask.imshow(overlay)
        ax_mask.set_title(
            f"Predicted Burn Mask\n({result['stats']['burned_fraction'] * 100:.1f}% burned)",
            fontsize=11,
        )
        ax_mask.axis("off")

        col_titles = ["RGB", "NDVI", "NBR"]
        for row, period_name in enumerate(["pre", "post"]):
            period = result[period_name]
            label = f"{period_name}\n({period['date']})"

            ax_rgb = fig.add_subplot(gs[row, 1])
            ax_rgb.imshow(normalize_rgb(period["rgb"]))
            ax_rgb.set_title(f"{col_titles[0]}\n{label}", fontsize=10)
            ax_rgb.axis("off")

            ax_ndvi = fig.add_subplot(gs[row, 2])
            im = ax_ndvi.imshow(period["ndvi"], cmap=index_cmap, norm=index_norm)
            ax_ndvi.set_title(f"{col_titles[1]}\n{label}", fontsize=10)
            ax_ndvi.axis("off")
            plt.colorbar(im, ax=ax_ndvi, fraction=0.046, pad=0.04)

            ax_nbr = fig.add_subplot(gs[row, 3])
            im = ax_nbr.imshow(period["nbr"], cmap=index_cmap, norm=index_norm)
            ax_nbr.set_title(f"{col_titles[2]}\n{label}", fontsize=10)
            ax_nbr.axis("off")
            plt.colorbar(im, ax=ax_nbr, fraction=0.046, pad=0.04)
    else:
        post = result["post"]
        gt_mask = result["ground_truth_mask"]
        n_cols = 5 if gt_mask is not None else 4
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        axes[0].imshow(normalize_rgb(post["rgb"]))
        axes[0].set_title("RGB", fontsize=10)
        im = axes[1].imshow(post["ndvi"], cmap=index_cmap, norm=index_norm)
        axes[1].set_title("NDVI", fontsize=10)
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        im = axes[2].imshow(post["nbr"], cmap=index_cmap, norm=index_norm)
        axes[2].set_title("NBR", fontsize=10)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        pred_overlay = overlay_burn_mask(post["rgb"], result["mask"])
        axes[3].imshow(pred_overlay)
        axes[3].set_title(
            f"Predicted Burn Mask\n({result['stats']['burned_fraction'] * 100:.1f}% burned)", fontsize=10
        )
        if gt_mask is not None:
            gt_overlay = overlay_burn_mask(post["rgb"], gt_mask, color=(0.0, 0.6, 1.0))
            axes[4].imshow(gt_overlay)
            axes[4].set_title(f"Ground Truth Mask\n({gt_mask.mean() * 100:.1f}% burned)", fontsize=10)
        for ax in axes:
            ax.axis("off")

    plt.suptitle(f"Pipeline Result — {result['mode']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    print("Running pipeline…", file=sys.stderr)
    result = run_pipeline(
        lat=args.lat,
        lon=args.lon,
        event_date=args.event_date,
        local_tif=args.local_tif,
        mask_threshold=args.mask_threshold,
        keywords=args.keywords,
        attribution_mode=args.attribution_mode,
        domain_prompt=args.domain_prompt,
        output_contract=args.output_contract,
    )
    summary = summarize_result(result)

    human_readable_stream = sys.stdout if args.output == "png" else sys.stderr
    print(render_human_readable(summary), file=human_readable_stream)

    if args.output in ("json", "both"):
        print(json.dumps(summary, indent=2))

    if args.output in ("png", "both"):
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{_slug(args)}_pipeline.png")
        _save_png(result, out_path)
        print(f"Saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
