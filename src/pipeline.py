"""
End-to-end orchestration: retrieval -> segmentation -> severity (-> attribution stub),
composed exactly as documented in src/segmentation/predict.py and src/severity/predict.py's
docstrings, but not previously wired together anywhere in the repo.

Importing this module (for fetch_reference_imagery) triggers gee_client.py's module-level
load_dotenv() as a side effect, so .env is loaded before run_pipeline is ever called, even
in local-file-only invocations.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from .attribution.predict import assess_attribution, skipped_attribution
from .retrieval.gee_client import SCALE_M
from .retrieval.local_source import HLS_PIXEL_SIZE_M, read_local_mask, read_local_scene
from .retrieval.reference_retrieval import POST_EVENT_DAYS, PRE_EVENT_DAYS, fetch_reference_imagery
from .segmentation.predict import predict_mask
from .severity.features import extract_features_from_indices
from .severity.predict import classify_severity, classify_severity_from_retrieval

SEVERITY_LABELS = {0: "none", 1: "minimal", 2: "moderate", 3: "maximal"}


def _iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    pred_bin = pred_mask > threshold
    gt_bin = gt_mask > threshold
    union = (pred_bin | gt_bin).sum()
    if union == 0:
        return 1.0
    return float((pred_bin & gt_bin).sum() / union)


def _pixel_ground_size_m(lat: float | None) -> tuple[float, float]:
    """
    (north-south, east-west) ground size of one patch pixel, in metres.

    GEE mode: gee_client reprojects to EPSG:4326 at `scale=SCALE_M`, which GEE reads as metres at
    the equator — so a pixel is a fixed SCALE_M/111320 degrees on both axes. North-south that is
    SCALE_M on the ground; east-west it shrinks by cos(lat) (~3% at El-Fasher's 13.8 N).
    Local mode: an HLS chip, fixed at HLS_PIXEL_SIZE_M.
    """
    if lat is None:
        return HLS_PIXEL_SIZE_M, HLS_PIXEL_SIZE_M
    return float(SCALE_M), float(SCALE_M) * math.cos(math.radians(lat))


def _scene_provenance(event_date: str, pre: dict, post: dict) -> dict:
    """
    What imagery the verdict actually rests on: the requested search windows, every Sentinel-2
    acquisition composited into each mosaic, and how far the post-event mosaic sits from the
    submitted date.

    `post_lag_days` is measured to the *latest* acquisition in the post mosaic, not the
    representative one, because mosaic() composites them all. A large lag matters: a burn scar in
    Sahel cropland fades within weeks, so a mosaic drawn long after the fire can be scar-free
    while the fire was perfectly real.

    No flag here can detect a *wrong* submitted event_date — the pre window ends the day before it
    by construction, so the pre mosaic is always "before" whatever date was supplied, even when
    that date is months after the true event. Only the operator can catch that, which is why the
    windows are printed rather than silently trusted.
    """
    event = datetime.strptime(event_date, "%Y-%m-%d")
    post_dates = [datetime.strptime(s["date"], "%Y-%m-%d") for s in post["scenes"]]
    pre_dates = [datetime.strptime(s["date"], "%Y-%m-%d") for s in pre["scenes"]]
    return {
        "event_date": event_date,
        "pre_window": [
            (event - timedelta(days=PRE_EVENT_DAYS)).strftime("%Y-%m-%d"),
            (event - timedelta(days=1)).strftime("%Y-%m-%d"),
        ],
        "post_window": [event_date, (event + timedelta(days=POST_EVENT_DAYS)).strftime("%Y-%m-%d")],
        "pre_scenes": pre["scenes"],
        "post_scenes": post["scenes"],
        "pre_lead_days": (event - min(pre_dates)).days,
        "post_lag_days": (max(post_dates) - event).days,
    }


def _mask_offsets_km(mask: np.ndarray, lat: float) -> tuple[float | None, float | None]:
    """
    (nearest, centroid) distance in km from the patch centre to the predicted mask, or
    (None, None) if the mask is empty.

    GEE mode only: _make_aoi buffers the supplied point, so the patch centre *is* the submitted
    lat/lon and this reduces to index arithmetic — no reprojection. This is what distinguishes
    "the model fired on the village" from "the model fired somewhere in a 399 km2 box".
    """
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None, None
    ns_m, ew_m = _pixel_ground_size_m(lat)
    height, width = mask.shape
    dy = (ys - (height - 1) / 2.0) * ns_m
    dx = (xs - (width - 1) / 2.0) * ew_m
    nearest_m = float(np.hypot(dx, dy).min())
    centroid_m = math.hypot(float(dx.mean()), float(dy.mean()))
    return nearest_m / 1000.0, centroid_m / 1000.0


def run_pipeline(
    lat: float | None = None,
    lon: float | None = None,
    event_date: str | None = None,
    local_tif: str | None = None,
    mask_threshold: float = 0.5,
    keywords: str | None = None,
    attribution_mode: str = "yes_no",
    domain_prompt: str | None = None,
    output_contract: str | None = None,
    news_radius_km: int | None = None,
    skip_attribution: bool = False,
) -> dict:
    """
    Run the full retrieval -> segmentation -> severity -> attribution(stub) pipeline.

    Exactly one of `local_tif` or `(lat, lon, event_date)` must be given.

    Args:
        lat, lon, event_date: GEE mode — see fetch_reference_imagery.
        local_tif: local mode — path to a local HLS *_merged.tif, bypassing GEE
            entirely (e.g. dataset/validation/<id>_merged.tif). If a sibling
            <id>.mask.tif exists, it's included as ground-truth for comparison.
        mask_threshold: sigmoid threshold passed to predict_mask.
        keywords, attribution_mode, domain_prompt, output_contract, news_radius_km: forwarded to
            assess_attribution() to override its World News search keywords / classification
            mode ("yes_no" or "structured") / Gemini classifier instructions / geographic search
            radius; None keeps their built-in defaults. Unused in local mode, where attribution
            is skipped.
        skip_attribution: run retrieval/segmentation/severity only and return the same
            "skipped"-shaped attribution block as local mode. For bulk evaluation sweeps
            (scripts/eval_yale_sites.py), where one Gemini + World News call per site per year
            would exhaust both quotas for no analytical gain.

    Returns:
        A dict with pre/post imagery (None pre in local mode), the predicted mask,
        severity classification, NDVI/NBR stats, scene provenance, and an attribution
        block. Numpy arrays included — use summarize_result() for a JSON-safe projection.

        "severity_basis" is "classifier" normally, or "empty_mask" when the segmenter found
        nothing and severity was reported as 0 without consulting the classifier — in that
        case "probabilities" is None.

        stats["burned_fraction"]/["mask_pixels"]/["burned_area_ha"] are measured directly from
        the predicted mask, NOT taken from extract_features_from_indices (whose fifth element
        carries a model-input fallback — see src/severity/features.py).

    Raises:
        ValueError: if both/neither of local_tif and (lat, lon, event_date) are
            given, or if no usable scene is found (GEE mode).
        EnvironmentError: if GEE_PROJECT isn't set (GEE mode), or if WORLD_NEWS_API_KEY /
            GEMINI_API_KEY isn't set (GEE mode — attribution needs lat/lon/event_date).
    """
    gee_mode = lat is not None or lon is not None or event_date is not None
    if local_tif is not None and gee_mode:
        raise ValueError("Provide either local_tif or (lat, lon, event_date), not both.")
    if local_tif is None and not gee_mode:
        raise ValueError("Provide either local_tif or (lat, lon, event_date).")
    if gee_mode and (lat is None or lon is None or event_date is None):
        raise ValueError("GEE mode requires all of lat, lon, and event_date.")

    if local_tif is not None:
        mode = "local"
        pipeline_input = {"local_tif": local_tif}
        pre = None
        post = read_local_scene(local_tif)
        mask = predict_mask(post["bands"], threshold=mask_threshold)
        ground_truth_mask = read_local_mask(local_tif)
        iou = _iou(mask, ground_truth_mask) if ground_truth_mask is not None else None
        pre_ndvi_mean = pre_nbr_mean = dndvi = dnbr = None
        retrieval_result = None
        windows = None
    else:
        mode = "gee"
        pipeline_input = {"lat": lat, "lon": lon, "event_date": event_date}
        retrieval_result = fetch_reference_imagery(lat, lon, event_date)
        pre = retrieval_result["pre"]
        post = retrieval_result["post"]
        mask = predict_mask(post["bands"], threshold=mask_threshold)
        ground_truth_mask = None
        iou = None
        pre_ndvi_mean, post_ndvi_mean = float(pre["ndvi"].mean()), float(post["ndvi"].mean())
        dndvi = pre_ndvi_mean - post_ndvi_mean
        pre_nbr_mean, post_nbr_mean = float(pre["nbr"].mean()), float(post["nbr"].mean())
        dnbr = pre_nbr_mean - post_nbr_mean
        windows = _scene_provenance(event_date, pre, post)

    # Measured extent, straight off the mask — deliberately NOT the fifth element of
    # extract_features_from_indices. That one carries a 1.0 fallback for an empty mask (see
    # severity/features.py) which is a model-input convention, not a measurement; reporting it
    # as "burn-affected area" is what produced "100% burned, maximal severity" on scenes whose
    # displayed mask was blank.
    mask_pixels = int((mask > 0).sum())
    burned_fraction = float(mask_pixels / mask.size)
    ns_m, ew_m = _pixel_ground_size_m(lat)
    burned_area_ha = mask_pixels * ns_m * ew_m / 10_000.0
    nearest_km, centroid_km = _mask_offsets_km(mask, lat) if mode == "gee" else (None, None)

    whole_ndvi, whole_nbr, masked_ndvi, masked_nbr, _feature_burned_fraction = extract_features_from_indices(
        post["ndvi"], post["nbr"], mask
    )

    if mask_pixels == 0:
        # Nothing was segmented, so there is no burn to grade. Running the classifier here would
        # feed it the whole-frame NDVI/NBR means plus a burned_fraction of 1.0 and reliably return
        # "maximal" — a confident verdict about a region the segmenter did not find.
        severity, probabilities, severity_basis = 0, None, "empty_mask"
    else:
        sev = (
            classify_severity(post["ndvi"], post["nbr"], mask=mask)
            if retrieval_result is None
            else classify_severity_from_retrieval(retrieval_result, mask=mask)
        )
        severity, probabilities, severity_basis = sev["severity"], sev["probabilities"], "classifier"

    if skip_attribution:
        attribution = skipped_attribution(
            attribution_mode, "Attribution skipped by request (skip_attribution=True)."
        )
    else:
        attribution = assess_attribution(
            lat, lon, event_date,
            keywords=keywords, mode=attribution_mode, domain_prompt=domain_prompt,
            output_contract=output_contract, radius_km=news_radius_km,
        )

    return {
        "mode": mode,
        "input": pipeline_input,
        "pre": pre,
        "post": post,
        "mask": mask,
        "ground_truth_mask": ground_truth_mask,
        "severity": severity,
        "severity_basis": severity_basis,
        "probabilities": probabilities,
        "windows": windows,
        "stats": {
            "whole_ndvi": whole_ndvi,
            "whole_nbr": whole_nbr,
            "masked_ndvi": masked_ndvi,
            "masked_nbr": masked_nbr,
            "burned_fraction": burned_fraction,
            "mask_pixels": mask_pixels,
            "burned_area_ha": burned_area_ha,
            "pixel_size_m": round(ns_m, 1),
            "patch_extent_km": round(mask.shape[0] * ns_m / 1000.0, 2),
            "nearest_mask_pixel_km": nearest_km,
            "mask_centroid_offset_km": centroid_km,
            "pre_ndvi_mean": pre_ndvi_mean,
            "post_ndvi_mean": float(post["ndvi"].mean()),
            "dndvi": dndvi,
            "pre_nbr_mean": pre_nbr_mean,
            "post_nbr_mean": float(post["nbr"].mean()),
            "dnbr": dnbr,
            "iou_vs_ground_truth": iou,
        },
        "attribution": attribution,
    }


def summarize_result(result: dict) -> dict:
    """Strip all ndarrays from a run_pipeline() result, keeping only JSON-serializable fields."""

    def _period(period: dict | None) -> dict | None:
        if period is None:
            return None
        return {"date": period.get("date"), "scenes": period.get("scenes")}

    return {
        "mode": result["mode"],
        "input": result["input"],
        "pre": _period(result["pre"]),
        "post": _period(result["post"]),
        "has_ground_truth": result["ground_truth_mask"] is not None,
        "severity": result["severity"],
        "severity_label": SEVERITY_LABELS[result["severity"]],
        "severity_basis": result["severity_basis"],
        "probabilities": result["probabilities"],
        "windows": result["windows"],
        "stats": result["stats"],
        "attribution": result["attribution"],
    }


def _fmt_mosaic(period: dict) -> str:
    """"2024-06-14" for a single acquisition, "2024-06-14 (+3 more, 2024-06-14..2024-06-29)" for a
    composite — so a mosaic spanning weeks never reads as one dated image."""
    scenes = period.get("scenes") or []
    if len(scenes) <= 1:
        return period["date"]
    dates = sorted(s["date"] for s in scenes)
    return f"{period['date']} (+{len(scenes) - 1} more, {dates[0]}..{dates[-1]})"


def render_human_readable(summary: dict, include_articles: bool = True) -> str:
    """
    Format a summarize_result()-shaped dict as human-readable text (used by CLI and web).

    include_articles: whether to list attribution's yes_list/no_list article titles/URLs as
        text bullets. The web UI renders those as an HTML table instead, so it passes False
        here to avoid showing the same articles twice.
    """
    stats = summary["stats"]
    lines = []
    lines.append("Environmental Warfare Classifier — Pipeline Result")
    lines.append("=" * 52)

    if summary["mode"] == "gee":
        inp = summary["input"]
        windows = summary["windows"]
        lines.append(f"Mode: GEE (lat={inp['lat']}, lon={inp['lon']}, event_date={inp['event_date']})")
        lines.append(
            f"Pre-event mosaic:  {_fmt_mosaic(summary['pre'])}  "
            f"[searched {windows['pre_window'][0]} .. {windows['pre_window'][1]}]"
        )
        lines.append(
            f"Post-event mosaic: {_fmt_mosaic(summary['post'])}  "
            f"[searched {windows['post_window'][0]} .. {windows['post_window'][1]}]"
        )
        if windows["post_lag_days"] > 14:
            lines.append(
                f"  ! The post-event mosaic reaches {windows['post_lag_days']} days past the given "
                "date. Burn scars in dry cropland can fade within weeks — a clean result here is "
                "weak evidence of no fire."
            )
    else:
        lines.append(f"Mode: local file ({summary['input']['local_tif']})")

    lines.append("")
    if summary["severity_basis"] == "empty_mask":
        lines.append("Severity: not assessed — the segmenter found no burn pixels in this patch.")
    else:
        lines.append(f"Severity: {summary['severity']} ({summary['severity_label']})")
        proba_str = "  ".join(f"{k}={v:.2f}" for k, v in sorted(summary["probabilities"].items()))
        lines.append(f"  Probabilities: {proba_str}")

    lines.append("")
    lines.append(
        f"Burn-affected area: {stats['burned_fraction'] * 100:.1f}% of patch "
        f"({stats['mask_pixels']:,} px, {stats['burned_area_ha']:,.0f} ha)"
    )
    lines.append(
        f"  Patch footprint: {stats['patch_extent_km']} km across "
        f"at {stats['pixel_size_m']} m/px"
    )
    if summary["mode"] == "gee":
        if stats["nearest_mask_pixel_km"] is None:
            lines.append("  Distance from the submitted point: n/a — no mask pixels")
        else:
            lines.append(
                f"  Nearest mask pixel is {stats['nearest_mask_pixel_km']:.2f} km from the "
                f"submitted point; mask centroid {stats['mask_centroid_offset_km']:.2f} km away"
            )

    if stats["dndvi"] is not None:
        verdict = "consistent with vegetation loss" if stats["dndvi"] > 0.05 else "no strong signal"
        lines.append(
            f"NDVI: pre={stats['pre_ndvi_mean']:.3f} post={stats['post_ndvi_mean']:.3f} "
            f"(Δ {-stats['dndvi']:.3f} — {verdict})"
        )
        verdict = "consistent with burn/vegetation loss" if stats["dnbr"] > 0.1 else "no strong burn signal"
        lines.append(
            f"NBR:  pre={stats['pre_nbr_mean']:.3f} post={stats['post_nbr_mean']:.3f} "
            f"(Δ {-stats['dnbr']:.3f} — {verdict})"
        )
    else:
        lines.append(f"NDVI (post-event): {stats['post_ndvi_mean']:.3f}")
        lines.append(f"NBR  (post-event): {stats['post_nbr_mean']:.3f}")

    if stats["iou_vs_ground_truth"] is not None:
        lines.append(f"IoU vs. ground truth: {stats['iou_vs_ground_truth']:.3f}")

    lines.append("")
    attribution = summary["attribution"]
    lines.append(f"Attribution: {attribution['note']}")
    if include_articles:
        structured = attribution.get("mode") == "structured"

        def _fmt_article(a: dict) -> list[str]:
            entry_lines = [f"    - {a['title']} — {a['url']}"]
            if structured:
                entry_lines.append(
                    f"        conflict_activity={a.get('conflict_activity')}  "
                    f"environmental_destruction={a.get('environmental_destruction')}  "
                    f"deliberate={a.get('deliberate')}  confidence={a.get('confidence')}"
                )
                entry_lines.append(f"        actor_claimed: {a.get('actor_claimed')}")
                entry_lines.append(f"        evidence: \"{a.get('evidence_quote')}\"")
            return entry_lines

        if attribution["yes_list"]:
            lines.append("  Supporting articles:")
            for a in attribution["yes_list"]:
                lines.extend(_fmt_article(a))
        if attribution["no_list"]:
            lines.append("  Opposing articles:")
            for a in attribution["no_list"]:
                lines.extend(_fmt_article(a))

    return "\n".join(lines)
