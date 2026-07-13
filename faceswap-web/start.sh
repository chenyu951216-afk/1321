#!/bin/sh
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR=${DATA_DIR:-/data}
MODEL_DIR=${MODEL_DIR:-/models}

export DATA_DIR MODEL_DIR

mkdir -p \
    "$DATA_DIR/faces" \
    "$DATA_DIR/temp" \
    "$DATA_DIR/results" \
    "$MODEL_DIR"

python "$APP_DIR/scripts/download_models.py" \
    --model-dir "$MODEL_DIR" \
    --allow-failure

cd "$APP_DIR"
exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}"
