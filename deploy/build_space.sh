#!/usr/bin/env bash
# Stage exactly what the HF Space needs into .space_build/, optionally uploading it.
#
#   deploy/build_space.sh                          # stage + print the file list, review it
#   deploy/build_space.sh --push <user>/<space>    # stage + upload, pruning stale files
#
# Explicit allowlist, never `cp -r src/`: the Space is public, and a file pushed once stays in
# its git history even after it's deleted. Keys (.env, ew-classifier-sa.json), dataset/,
# checkpoints/ and training code never enter the staging tree. Weights live in a separate
# model repo (see deploy/app.py).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$ROOT/.space_build"

rm -rf "$STAGE"
mkdir -p "$STAGE"

# The Space's own files, at its root.
cp "$ROOT/deploy/app.py" "$ROOT/deploy/requirements.txt" "$ROOT/deploy/README.md" "$STAGE/"

# Web UI (routes reused by deploy/app.py, and the page).
mkdir -p "$STAGE/web/static"
cp "$ROOT/web/__init__.py" "$ROOT/web/app.py" "$STAGE/web/"
cp "$ROOT/web/static/index.html" "$STAGE/web/static/"

# Only the src/ modules the inference pipeline imports.
SRC_FILES=(
  __init__.py
  pipeline.py
  visualization.py
  retrieval/__init__.py
  retrieval/gee_client.py
  retrieval/local_source.py
  retrieval/reference_retrieval.py
  segmentation/__init__.py
  segmentation/dataset.py
  segmentation/model.py
  segmentation/predict.py
  severity/__init__.py
  severity/features.py
  severity/model.py
  severity/predict.py
  attribution/__init__.py
  attribution/dedup.py
  attribution/gemini_client.py
  attribution/predict.py
  attribution/worldnews_client.py
)
for f in "${SRC_FILES[@]}"; do
  mkdir -p "$STAGE/src/$(dirname "$f")"
  cp "$ROOT/src/$f" "$STAGE/src/$f"
done

echo "Staged $(du -sh "$STAGE" | cut -f1) into $STAGE:"
(cd "$STAGE" && find . -type f | sed 's|^\./|  |' | sort)

if [[ "${1:-}" == "--push" ]]; then
  SPACE="${2:?usage: $0 --push <user>/<space>}"
  # upload_folder rather than `hf upload`: the CLI first calls create_repo, which a free account
  # gets a 402 for (it asks for cpu-basic) even when the ZeroGPU Space already exists. The Space
  # must be created beforehand with space_hardware="zero-a10g".
  # delete_patterns prunes files a previous deploy shipped that this staging tree no longer has.
  python - "$SPACE" "$STAGE" "$(git -C "$ROOT" rev-parse --short HEAD)" <<'PY'
import sys
from huggingface_hub import upload_folder

space, stage, rev = sys.argv[1:]
commit = upload_folder(
    repo_id=space,
    repo_type="space",
    folder_path=stage,
    delete_patterns="*",
    commit_message=f"Deploy from {rev}",
)
print(commit)
PY
fi
