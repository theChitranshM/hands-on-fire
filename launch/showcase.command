#!/bin/zsh
# Double-click to run in Terminal. (Terminal can ask for camera permission; IDE/AI-app terminals often cannot.)
cd "$(dirname "$0")/.." && .venv/bin/python run.py --showcase
echo; read -k1 "?Done. Press any key to close."
