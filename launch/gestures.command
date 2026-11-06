#!/bin/zsh
# Double-click to run in Terminal. (Terminal can ask for camera permission; IDE/AI-app terminals often cannot.)
cd "$(dirname "$0")/.." && .venv/bin/python tools/gesture_debug.py
echo; read -k1 "?Done. Press any key to close."
