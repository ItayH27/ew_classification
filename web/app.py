"""
Local FastAPI web interface for the pipeline (src/pipeline.py).

Run from the repo root, inside the ew_DL conda env:
    python -m uvicorn web.app:app --host 0.0.0.0 --port 8000

The `-m` form matters: it's what puts the repo root on sys.path so `import web.app`
(and this module's own `from src.pipeline import ...`) work, unlike the bare
`uvicorn` console script, which does not add the current working directory.

Single worker only: src/segmentation/predict.py and src/severity/predict.py each
cache their loaded model in a module-level singleton, so extra uvicorn workers would
just duplicate model memory per process for no throughput benefit at this scale.
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.attribution.gemini_client import _STRUCTURED_FORMAT_INSTRUCTION as ATTRIBUTION_STRUCTURED_FORMAT_INSTRUCTION
from src.attribution.gemini_client import get_mode_defaults as get_attribution_mode_defaults
from src.attribution.worldnews_client import DEFAULT_RADIUS_KM as DEFAULT_ATTRIBUTION_RADIUS_KM
from src.attribution.worldnews_client import _DEFAULT_KEYWORDS as DEFAULT_ATTRIBUTION_KEYWORDS
from src.pipeline import render_human_readable, run_pipeline, summarize_result
from src.segmentation.predict import predict_mask
from src.severity.predict import classify_severity
from src.visualization import array_to_png_base64, normalize_rgb, overlay_burn_mask

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm both models' module-level singleton caches so the first real request isn't slow.
    predict_mask(np.zeros((256, 256, 6), dtype=np.float32))
    classify_severity(np.zeros((256, 256), dtype=np.float32), np.zeros((256, 256), dtype=np.float32))
    yield


app = FastAPI(title="Environmental Warfare Classifier", lifespan=lifespan)


def _build_images(result: dict) -> dict:
    """
    Key order matches the intended layout: GEE mode puts the predicted mask first (rendered
    on the left, spanning both rows) followed by pre/post RGB-NDVI-NBR pairs; local mode keeps
    its existing RGB/NDVI/NBR/predicted-mask/ground-truth-mask row order.
    """
    post = result["post"]
    post_mask_overlay = array_to_png_base64(overlay_burn_mask(post["rgb"], result["mask"]))
    post_rgb = array_to_png_base64(normalize_rgb(post["rgb"]))
    post_ndvi = array_to_png_base64(post["ndvi"], cmap="RdYlGn", vmin=-1, vmax=1)
    post_nbr = array_to_png_base64(post["nbr"], cmap="RdYlGn", vmin=-1, vmax=1)

    if result["pre"] is not None:
        pre = result["pre"]
        return {
            "post_mask_overlay": post_mask_overlay,
            "pre_rgb": array_to_png_base64(normalize_rgb(pre["rgb"])),
            "pre_ndvi": array_to_png_base64(pre["ndvi"], cmap="RdYlGn", vmin=-1, vmax=1),
            "pre_nbr": array_to_png_base64(pre["nbr"], cmap="RdYlGn", vmin=-1, vmax=1),
            "post_rgb": post_rgb,
            "post_ndvi": post_ndvi,
            "post_nbr": post_nbr,
        }

    images = {
        "post_rgb": post_rgb,
        "post_ndvi": post_ndvi,
        "post_nbr": post_nbr,
        "post_mask_overlay": post_mask_overlay,
    }
    if result["ground_truth_mask"] is not None:
        images["ground_truth_overlay"] = array_to_png_base64(
            overlay_burn_mask(post["rgb"], result["ground_truth_mask"], color=(0.0, 0.6, 1.0))
        )
    return images


def _respond(result: dict) -> dict:
    summary = summarize_result(result)
    return {
        **summary,
        "human_readable": render_human_readable(summary, include_articles=False),
        "images": _build_images(result),
    }


class RunRequest(BaseModel):
    lat: float
    lon: float
    event_date: str
    keywords: str | None = None
    # Geographic filter on the news search. The commonest reason a rural location returns zero
    # articles is that nothing is geotagged within the default radius, so it is worth widening.
    news_radius_km: int | None = None
    attribution_mode: Literal["yes_no", "structured"] = "yes_no"
    domain_prompt: str | None = None
    output_contract: str | None = None


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/config")
def api_config():
    """Attribution defaults, so the web UI can show/edit them without duplicating the values."""
    yes_no_domain_prompt, yes_no_output_contract = get_attribution_mode_defaults("yes_no")
    structured_domain_prompt, structured_output_contract = get_attribution_mode_defaults("structured")
    return {
        "keywords": DEFAULT_ATTRIBUTION_KEYWORDS,
        "news_radius_km": DEFAULT_ATTRIBUTION_RADIUS_KM,
        "modes": {
            "yes_no": {"domain_prompt": yes_no_domain_prompt, "output_contract": yes_no_output_contract},
            "structured": {
                "domain_prompt": structured_domain_prompt,
                "output_contract": structured_output_contract,
                # Fixed, non-overridable — see gemini_client._STRUCTURED_FORMAT_INSTRUCTION. Surfaced
                # here purely for display, so the web UI can show what's always appended to
                # output_contract regardless of edits, rather than leaving it invisible.
                "format_instruction": ATTRIBUTION_STRUCTURED_FORMAT_INSTRUCTION,
            },
        },
    }


@app.post("/api/run")
def api_run(req: RunRequest):
    """GEE mode: fetch pre/post Sentinel-2 imagery for a lat/lon + event date and run the pipeline."""
    try:
        result = run_pipeline(
            lat=req.lat,
            lon=req.lon,
            event_date=req.event_date,
            keywords=req.keywords,
            news_radius_km=req.news_radius_km,
            attribution_mode=req.attribution_mode,
            domain_prompt=req.domain_prompt,
            output_contract=req.output_contract,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _respond(result)


@app.post("/api/run_local")
def api_run_local(file: UploadFile = File(...)):
    """Local mode: run the pipeline directly on an uploaded HLS *_merged.tif, bypassing GEE."""
    with tempfile.NamedTemporaryFile(suffix="_merged.tif") as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        try:
            result = run_pipeline(local_tif=tmp.name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return _respond(result)
