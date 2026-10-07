#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
echo "Setup complete. Next: bash run_all.sh"
