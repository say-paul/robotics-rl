#!/bin/bash
# One command to start VNC + MuJoCo viewer.
#
# Usage:
#   bash vnc/start_vnc.sh                                    # scene only
#   bash vnc/start_vnc.sh --policy policies/g1_stand_v3.onnx # with policy
#
# Then open:  http://<your-ip>:6080/vnc.html
# Or connect: <your-ip>:5900 with any VNC client
set -e
cd "$(dirname "$0")/.."

IMAGE=g1-vnc

cleanup() {
    kill "$VIEWER_PID" 2>/dev/null
    podman stop "$IMAGE" 2>/dev/null
    podman rm -f "$IMAGE" 2>/dev/null
}
trap cleanup EXIT

podman build -t "$IMAGE" -f vnc/Containerfile vnc/
podman rm -f "$IMAGE" 2>/dev/null || true
podman run -d --name "$IMAGE" --network=host "$IMAGE"

sleep 3
echo "==> Waiting for VNC server..."
for i in $(seq 1 10); do
    python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',5900)); s.close()" 2>/dev/null && break
    sleep 1
done

echo "==> Launching MuJoCo viewer..."
DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 python3 vnc/sim_viewer.py "$@" &
VIEWER_PID=$!

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "============================================"
echo "  Browser:     http://$IP:6080/vnc.html"
echo "  VNC client:  $IP:5900"
echo "  Ctrl+C to stop"
echo "============================================"
echo ""
wait "$VIEWER_PID"
