#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
echo "Setup complete. Next: bash run_all.sh"
