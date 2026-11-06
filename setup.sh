#!/usr/bin/env bash
# One-command setup: venv, deps, hand model, sounds, device + permission check.
set -euo pipefail
cd "$(dirname "$0")"
[ "$(uname -m)" = "arm64" ] || echo "!! Intel Mac detected: works, but expect lower tracking fps."
[ "$(uname -s)" = "Darwin" ] || { echo "macOS only (uses AVFoundation cameras + macOS displays)."; exit 1; }
PY=$(command -v python3.11 || command -v python3.12 || true)
[ -n "$PY" ] || { echo "Python 3.11 or 3.12 needed. Install: brew install python@3.11"; exit 1; }
echo "Using $PY"
command -v ffmpeg >/dev/null || { echo "ffmpeg missing (needed for clips + camera names). Install: brew install ffmpeg"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
MODEL=assets/models/hand_landmarker.task
[ -f "$MODEL" ] || { mkdir -p assets/models; curl -sSL -o "$MODEL" \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task; }
.venv/bin/python -m tools.gen_sounds >/dev/null
mkdir -p calibration logs captures
echo
echo "== Devices =="
.venv/bin/python tools/list_devices.py || true
echo
echo "If the camera list is empty or a later step says permission denied:"
echo "  Run camera tools from the Terminal app (double-click launch/*.command) and click Allow."
echo
echo "Setup done. Try it without any hardware:  .venv/bin/python run.py --sim"
echo "Real show: double-click launch/calibrate.command, then launch/show.command (see README)."
