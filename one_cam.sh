#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/base_env"
REQ_FILE="$SCRIPT_DIR/requirements_base.txt"
ROS_SCRIPT="$SCRIPT_DIR/ros_publisher_single.py"

cd "$SCRIPT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
else
    source "$VENV_DIR/bin/activate"
fi

python "$ROS_SCRIPT"