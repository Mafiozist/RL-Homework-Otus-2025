# scripts/setup.sh
#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
python -m ipykernel install --user --name rl-homework --display-name "Python (rl-homework)"