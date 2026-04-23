#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/hsv_env"
REQ_FILE="$SCRIPT_DIR/requirements_hsv.txt"
TUNE_SCRIPT="$SCRIPT_DIR/hsv_tune.py"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_models.py"

# Set your token here for the session (or add to your ~/.bashrc)

cd "$SCRIPT_DIR"

# 1. Virtual Env Setup
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install -r "$REQ_FILE"
fi

# 2. Run the Python Downloader
python "$DOWNLOAD_SCRIPT"

# 3. Run Your Main Script
python "$TUNE_SCRIPT"