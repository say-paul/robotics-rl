#!/bin/bash
# Minimal install for MuJoCo simulation only.
# For full robot setup (CycloneDDS, Unitree SDK, WBC), run: ./setup_robot.sh
set -e

echo "=== RDP Install (simulation) ==="

python3 -m venv venv 2>/dev/null || true
source venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "Done. Activate with:  source venv/bin/activate"
echo "Then run:  python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml"
echo ""
echo "For full robot setup (CycloneDDS + Unitree SDK + WBC):  ./setup_robot.sh"
