#!/bin/bash
# Minimal install for MuJoCo simulation only.
# For full robot setup (CycloneDDS, Unitree SDK, WBC), run: ./setup_robot.sh
set -e

echo "=== RDP Install (simulation) ==="

python3 -m venv venv 2>/dev/null || true
source venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt

# Auto-detect NVIDIA GPU and install GPU-accelerated packages
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "NVIDIA GPU detected — installing GPU acceleration..."
    pip install -q onnxruntime-gpu nvidia-cudnn-cu12

    CUDNN_LIB="$VIRTUAL_ENV/lib/python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cudnn/lib"
    if [ -d "$CUDNN_LIB" ] && ! grep -q "nvidia/cudnn" "$VIRTUAL_ENV/bin/activate" 2>/dev/null; then
        echo "export LD_LIBRARY_PATH=\"$CUDNN_LIB:\$LD_LIBRARY_PATH\"" >> "$VIRTUAL_ENV/bin/activate"
        echo "cuDNN library path added to venv activate script."
    fi
else
    echo ""
    echo "No NVIDIA GPU detected — using CPU inference."
fi

echo ""
echo "Done. Activate with:  source venv/bin/activate"
echo "Then run:  python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml"
echo ""
echo "For full robot setup (CycloneDDS + Unitree SDK + WBC):  ./setup_robot.sh"
