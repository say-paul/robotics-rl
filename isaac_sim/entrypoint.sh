#!/bin/bash
# This project was developed with assistance from AI tools.
set -e

export CYCLONEDDS_HOME=${CYCLONEDDS_HOME:-/opt/cyclonedds}

if [ ! -d "/groot/gear_sonic" ]; then
    echo "Error: mount GR00T-WholeBodyControl at /groot"
    exit 1
fi

# Create scene USD
echo "[entrypoint] Creating scene..."
/isaac-sim/python.sh /rdp/isaac_sim/create_g1_scene.py \
    --robot-usd /groot/gear_sonic/data/robots/g1/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd \
    --output /tmp/g1_scene.usd

echo "[entrypoint] Starting DDS bridge..."
exec /isaac-sim/python.sh /rdp/isaac_sim/dds_bridge.py \
    --usd /tmp/g1_scene.usd --robot-prim /World/G1 --domain-id 0
