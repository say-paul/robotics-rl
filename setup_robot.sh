#!/bin/bash
# Full robot setup: CycloneDDS, Unitree SDK2, GR00T WBC, and RDP.
# For simulation-only install, use: ./install.sh
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME"

echo "========================================="
echo "RDP Full Robot Setup"
echo "========================================="

# --- Python venv ---
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 not found. Install it first:"
    echo "  sudo apt install python3.12 python3.12-venv"
    exit 1
fi

VENV_DIR="venv"
VENV_RECREATED=false

if [ -d "$VENV_DIR" ]; then
    VENV_PY=$("$VENV_DIR/bin/python" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [[ "$VENV_PY" != "3.12" ]]; then
        echo "Existing venv is Python $VENV_PY, recreating with 3.12..."
        rm -rf "$VENV_DIR"
        python3.12 -m venv "$VENV_DIR"
        VENV_RECREATED=true
    fi
else
    python3.12 -m venv "$VENV_DIR"
    VENV_RECREATED=true
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# --- System deps ---
if command -v apt-get &> /dev/null; then
    echo ""
    echo "Installing system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-dev git git-lfs build-essential cmake libeigen3-dev
fi

# --- CycloneDDS ---
echo ""
echo "========================================="
echo "CycloneDDS"
echo "========================================="
CYCLONEDDS_DIR="$INSTALL_DIR/cyclonedds"

if [ "$VENV_RECREATED" = true ] && [ -d "$CYCLONEDDS_DIR" ]; then
    rm -rf "$CYCLONEDDS_DIR"
fi

if [ -d "$CYCLONEDDS_DIR/install" ] && { [ -f "$CYCLONEDDS_DIR/install/lib/libddsc.so" ] || [ -f "$CYCLONEDDS_DIR/install/lib64/libddsc.so" ]; }; then
    echo "Already installed at $CYCLONEDDS_DIR/install"
else
    [ -d "$CYCLONEDDS_DIR" ] && rm -rf "$CYCLONEDDS_DIR"
    cd "$INSTALL_DIR"
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
    cd cyclonedds && mkdir build install && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    cmake --build . --target install
    echo "Built and installed."
fi

export CYCLONEDDS_HOME="$CYCLONEDDS_DIR/install"

# Persist in venv activation
VENV_ACTIVATE="$PROJECT_DIR/$VENV_DIR/bin/activate"
if ! grep -q "CYCLONEDDS_HOME" "$VENV_ACTIVATE" 2>/dev/null; then
    echo "export CYCLONEDDS_HOME=\"$CYCLONEDDS_DIR/install\"" >> "$VENV_ACTIVATE"
fi

# --- Unitree SDK2 ---
echo ""
echo "========================================="
echo "unitree_sdk2_python"
echo "========================================="
UNITREE_SDK_DIR="$INSTALL_DIR/unitree_sdk2_python"

if python -c "import unitree_sdk2py" 2>/dev/null && [ "$VENV_RECREATED" = false ]; then
    echo "Already installed."
else
    if [ ! -d "$UNITREE_SDK_DIR" ]; then
        cd "$INSTALL_DIR"
        git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
    fi
    cd "$UNITREE_SDK_DIR"
    pip3 install -e .
    echo "Installed."
fi

# --- GR00T WBC ---
echo ""
echo "========================================="
echo "GR00T Whole-Body Control"
echo "========================================="
GROOT_WBC_DIR="$INSTALL_DIR/GR00T-WholeBodyControl"

if [ ! -d "$GROOT_WBC_DIR" ]; then
    echo "Cloning GR00T-WholeBodyControl..."
    cd "$INSTALL_DIR"
    git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
fi

echo "Which WBC backend?"
echo "  1) Decoupled WBC"
echo "  2) SONIC WBC"
echo "  3) Skip"
read -p "Choice [1-3]: " wbc_choice

case $wbc_choice in
    1)
        cd "$GROOT_WBC_DIR/decoupled_wbc"
        cp "$GROOT_WBC_DIR/README.md" . 2>/dev/null || true
        cp "$GROOT_WBC_DIR/LICENSE" . 2>/dev/null || true
        sed -i.bak 's|readme = "\.\./README\.md"|readme = "README.md"|' pyproject.toml
        sed -i.bak 's|license = {file = "\.\./LICENSE"}|license = {file = "LICENSE"}|' pyproject.toml
        pip install -e .
        mv pyproject.toml.bak pyproject.toml
        echo "Decoupled WBC installed."
        ;;
    2)
        cd "$GROOT_WBC_DIR/gear_sonic"
        cp "$GROOT_WBC_DIR/README.md" . 2>/dev/null || true
        cp "$GROOT_WBC_DIR/LICENSE" . 2>/dev/null || true
        sed -i.bak 's|readme = "\.\./README\.md"|readme = "README.md"|' pyproject.toml
        sed -i.bak 's|license = {file = "\.\./LICENSE"}|license = {file = "LICENSE"}|' pyproject.toml
        pip install -e .
        mv pyproject.toml.bak pyproject.toml
        echo "SONIC WBC installed."
        ;;
    *) echo "Skipped WBC install." ;;
esac

# --- RDP ---
cd "$PROJECT_DIR"
echo ""
echo "Installing RDP..."
pip install -r requirements.txt

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected — installing GPU acceleration..."
    pip install -q onnxruntime-gpu nvidia-cudnn-cu12

    CUDNN_LIB="$PROJECT_DIR/$VENV_DIR/lib/python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cudnn/lib"
    if [ -d "$CUDNN_LIB" ] && ! grep -q "nvidia/cudnn" "$VENV_ACTIVATE" 2>/dev/null; then
        echo "export LD_LIBRARY_PATH=\"$CUDNN_LIB:\$LD_LIBRARY_PATH\"" >> "$VENV_ACTIVATE"
        echo "cuDNN library path added to venv activate script."
    fi
fi
echo "Done."

# --- GR00T model ---
echo ""
read -p "Download GR00T N1.6 model? [y/N]: " dl
if [[ "$dl" =~ ^[Yy]$ ]]; then
    python scripts/download_groot_model.py --version n1.6 --verify
fi

# --- Verify ---
echo ""
echo "========================================="
echo "Verification"
echo "========================================="
python3 -c "
import os, sys
h = os.environ.get('CYCLONEDDS_HOME', '')
print(f'CYCLONEDDS_HOME: {h}' if h else 'CYCLONEDDS_HOME: not set')
try:
    import unitree_sdk2py; print('unitree_sdk2_python: ok')
except ImportError:
    print('unitree_sdk2_python: missing')
try:
    import decoupled_wbc; print('Decoupled WBC: ok')
except ImportError:
    pass
try:
    import gear_sonic; print('SONIC WBC: ok')
except ImportError:
    pass
import numpy, yaml, mujoco, onnxruntime
print('Core deps: ok')
"

echo ""
echo "========================================="
echo "Setup complete."
echo "========================================="
echo "Activate:  source venv/bin/activate"
echo "Launch:    python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml"
