"""
Hugging Face Space entry point (Gradio SDK on ZeroGPU) for the web UI in web/app.py.

Free HF Spaces only run the Gradio SDK, so this hosts web/app.py's existing FastAPI routes
and page inside gradio.Server (a FastAPI subclass) instead of a bare FastAPI app — same
endpoints, same index.html. Only the segmentation forward pass runs on a GPU slice (see
src/segmentation/predict.py's `spaces.GPU` wrapper).

Weights come from a separate HF model repo named by the MODEL_REPO Space variable, laid out as
    burn_scar/best_model/   (diffusers-format BurnScarSegmenter)
    severity/model.joblib
API keys (GEE_PROJECT, GEE_SERVICE_ACCOUNT, GEE_KEY_JSON, WORLD_NEWS_API_KEY, GEMINI_API_KEY)
are Space secrets, read from the environment by the src/ clients as usual.

Staged into a flat tree by deploy/build_space.sh. Also runs from a repo checkout:
    MODEL_REPO=<user>/ew-classifier-weights python deploy/app.py
or, with local weights and no download:
    EW_CHECKPOINT_ROOT=checkpoints python deploy/app.py
"""

import os
import sys
from pathlib import Path

# Staged Space: src/ and web/ sit beside this file. Repo checkout: they're one level up.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "src").is_dir() else _HERE.parent
sys.path.insert(0, str(ROOT))

_REQUIRED_WEIGHTS = ["burn_scar/best_model/config.json", "severity/model.joblib"]


def resolve_weights() -> str:
    """Download MODEL_REPO (cached) unless EW_CHECKPOINT_ROOT already points at local weights."""
    if os.environ.get("EW_CHECKPOINT_ROOT"):
        root = os.environ["EW_CHECKPOINT_ROOT"]
    else:
        model_repo = os.environ.get("MODEL_REPO")
        if not model_repo:
            raise EnvironmentError(
                "Set MODEL_REPO (a Space variable, e.g. <user>/ew-classifier-weights) "
                "or EW_CHECKPOINT_ROOT (a local checkpoints/ directory)."
            )
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo_id=model_repo, repo_type="model")

    missing = [p for p in _REQUIRED_WEIGHTS if not (Path(root) / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Weights at {root} are missing {missing}. Likely cause: the weights upload is "
            "incomplete, or was uploaded without the burn_scar/ and severity/ prefixes."
        )
    return root


# Must be set before importing src.*: the predict modules read it into their default
# checkpoint paths at import time.
os.environ["EW_CHECKPOINT_ROOT"] = resolve_weights()

from gradio import Server  # noqa: E402

from src.segmentation.predict import _get_model as _load_segmenter  # noqa: E402
from src.severity.predict import _get_model as _load_severity  # noqa: E402
from web import app as web_app  # noqa: E402

# Load both models at import, not per request, so the first visitor doesn't pay the loading
# cost. The segmenter stays on CPU until its spaces.GPU-wrapped forward pass (see
# src/segmentation/predict.py). (web_app's own lifespan warmup isn't used here — it would run
# a forward pass, i.e. claim a GPU slice.)
_load_segmenter()
_load_severity()

app = Server()
app.add_api_route("/", web_app.index, methods=["GET"])
app.add_api_route("/api/config", web_app.api_config, methods=["GET"])
app.add_api_route("/api/run", web_app.api_run, methods=["POST"])
app.add_api_route("/api/run_local", web_app.api_run_local, methods=["POST"])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
