#!/usr/bin/env python3
# This project was developed with assistance from AI tools.
"""Subscribe to teleimager's ZMQ camera stream and serve via HLS.

Connects to the Isaac Sim container's head_camera ZMQ publisher,
receives JPEG frames, and pipes them to ffmpeg for HLS streaming.
Open http://localhost:8090 in a browser to view.

Usage:
    python isaac_sim/zmq_to_hls.py [--host localhost] [--port 55555]
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import zmq

HLS_DIR = "/tmp/hls"
LISTEN_PORT = 8090
TARGET_FPS = 30

_frame_queue = queue.Queue(maxsize=4)

_INDEX_HTML = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>G1 Sim</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1e1e1e;color:#ccc;font:14px monospace;display:flex;flex-direction:column;height:100vh}
video{flex:1;object-fit:contain;background:#111}
#hud{position:fixed;top:10px;left:10px;color:#888}
#controls{display:flex;justify-content:center;gap:8px;padding:8px;background:#2a2a2a}
#controls .key{padding:8px 14px;background:#444;border:1px solid #666;border-radius:4px;cursor:pointer;user-select:none}
#controls .key:hover{background:#555}
#controls .key.active{background:#4a7;color:#fff}
#vel{position:fixed;bottom:50px;left:10px;color:#6f6}
</style></head>
<body>
<div id="hud">Connecting...</div>
<video id="v" autoplay muted playsinline></video>
<div id="vel">vx=0.0 vy=0.0 yaw=0.0</div>
<div id="controls">
<span class="key" data-key="w">W fwd</span>
<span class="key" data-key="s">S back</span>
<span class="key" data-key="a">A left</span>
<span class="key" data-key="d">D right</span>
<span class="key" data-key="q">Q strafe-L</span>
<span class="key" data-key="e">E strafe-R</span>
<span class="key" data-key="x">X stop</span>
</div>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<script>
const u='/hls/stream.m3u8',v=document.getElementById('v'),hud=document.getElementById('hud');
if(Hls.isSupported()){const h=new Hls({liveSyncDurationCount:2,liveMaxLatencyDurationCount:5,
liveDurationInfinity:true,lowLatencyMode:false,enableWorker:true,backBufferLength:0});
h.loadSource(u);h.attachMedia(v);h.on(Hls.Events.MANIFEST_PARSED,()=>{hud.textContent='Playing';v.play()});
h.on(Hls.Events.ERROR,(_,d)=>{hud.textContent='Error: '+d.details;if(d.fatal){
if(d.type===Hls.ErrorTypes.NETWORK_ERROR)h.startLoad();
else if(d.type===Hls.ErrorTypes.MEDIA_ERROR)h.recoverMediaError()}})}
else if(v.canPlayType('application/vnd.apple.mpegurl')){v.src=u;
v.addEventListener('playing',()=>{hud.textContent='Playing'})}

function send(key){
  fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:key})}).then(r=>r.json()).then(d=>{
    document.getElementById('vel').textContent=
      'vx='+d.vx.toFixed(1)+' vy='+d.vy.toFixed(1)+' yaw='+d.vyaw.toFixed(1)});
}
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if('wsadqex'.includes(k))send(k);
});
document.querySelectorAll('.key').forEach(el=>{
  el.addEventListener('click',()=>send(el.dataset.key));
});
</script></body></html>"""


_velocity = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "height": 0.8}
_shm_writer = None


def _init_shm(shm_name):
    """Connect to the sim's run_command shared memory."""
    global _shm_writer
    if shm_name:
        try:
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(name=shm_name)
            _shm_writer = shm
            print(f"[cmd] Connected to shared memory: {shm_name}")
        except Exception as e:
            print(f"[cmd] Shared memory not available: {e}")


def _write_shm(data_dict):
    """Write JSON data to shared memory in unitree_sim format."""
    if not _shm_writer:
        return
    import struct
    json_str = json.dumps(data_dict)
    json_bytes = json_str.encode("utf-8")
    timestamp = int(time.time()) & 0xFFFFFFFF
    buf = struct.pack("I", timestamp) + struct.pack("I", len(json_bytes)) + json_bytes
    _shm_writer.buf[:len(buf)] = buf


def _handle_key(key):
    v = _velocity
    if key == "w":
        v["vx"] = min(v["vx"] + 0.1, 1.0)
    elif key == "s":
        v["vx"] = max(v["vx"] - 0.1, -0.6)
    elif key == "a":
        v["vyaw"] = min(v["vyaw"] + 0.2, 1.57)
    elif key == "d":
        v["vyaw"] = max(v["vyaw"] - 0.2, -1.57)
    elif key == "q":
        v["vy"] = min(v["vy"] + 0.1, 0.5)
    elif key == "e":
        v["vy"] = max(v["vy"] - 0.1, -0.5)
    elif key == "x":
        v["vx"], v["vy"], v["vyaw"] = 0.0, 0.0, 0.0

    cmd = f'[{v["vx"]:.2f}, {v["vy"]:.2f}, {v["vyaw"]:.2f}, {v["height"]:.2f}]'
    _write_shm({"run_command": cmd})
    return v


class HlsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        if self.path == "/cmd":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            result = _handle_key(body.get("key", ""))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
        elif path.startswith("/hls/"):
            filename = path[5:]
            if "/" in filename or ".." in filename:
                self.send_response(403)
                self.end_headers()
                return
            filepath = os.path.join(HLS_DIR, filename)
            if not os.path.isfile(filepath):
                self.send_response(404)
                self.end_headers()
                return
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            ct = "application/vnd.apple.mpegurl" if filename.endswith(".m3u8") else "video/mp2t"
            self.send_header("Content-Type", ct)
            self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()
            self.wfile.write(data)
        elif path in ("/health", "/healthz"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()


def zmq_receiver(host, port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    addr = f"tcp://{host}:{port}"
    print(f"[zmq] Connecting to {addr}...")
    sock.connect(addr)

    count = 0
    while True:
        try:
            jpg_bytes = sock.recv()
            count += 1
            try:
                _frame_queue.put_nowait(jpg_bytes)
            except queue.Full:
                pass
            if count == 1:
                print(f"[zmq] First frame received ({len(jpg_bytes)} bytes)")
            elif count % 300 == 0:
                print(f"[zmq] {count} frames received")
        except Exception as e:
            print(f"[zmq] Error: {e}")
            time.sleep(1)


def ffmpeg_writer(cpu=False):
    os.makedirs(HLS_DIR, exist_ok=True)
    for f in os.listdir(HLS_DIR):
        os.remove(os.path.join(HLS_DIR, f))

    if cpu:
        encoder_args = ["-c:v", "libx264", "-preset", "ultrafast",
                        "-tune", "zerolatency", "-b:v", "2M"]
    else:
        encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4",
                        "-tune", "ll", "-b:v", "4M"]

    cmd = [
        "ffmpeg", "-y",
        "-use_wallclock_as_timestamps", "1",
        "-f", "image2pipe",
        "-codec:v", "mjpeg",
        "-framerate", str(TARGET_FPS),
        "-i", "pipe:0",
        *encoder_args,
        "-g", "30",
        "-keyint_min", "30",
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "8",
        "-hls_flags", "delete_segments+append_list+omit_endlist",
        "-hls_segment_type", "mpegts",
        os.path.join(HLS_DIR, "stream.m3u8"),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    threading.Thread(target=lambda: [print(f"[ffmpeg] {l.decode().rstrip()}") for l in proc.stderr], daemon=True).start()
    print("[hls] ffmpeg started (MJPEG -> HLS)")

    count = 0
    while True:
        jpg = _frame_queue.get()
        try:
            proc.stdin.write(jpg)
            proc.stdin.flush()
            count += 1
            if count == 1:
                print("[hls] First frame written")
        except (BrokenPipeError, OSError):
            print("[hls] ffmpeg pipe broken, restarting")
            proc.kill()
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description="ZMQ camera to HLS")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=55555)
    parser.add_argument("--cpu", action="store_true",
                        help="Use libx264 CPU encoder instead of NVENC")
    parser.add_argument("--shm", default=None,
                        help="Shared memory name for velocity commands")
    parser.add_argument("--shm-auto", action="store_true",
                        help="Auto-detect shared memory from /dev/shm")
    args = parser.parse_args()

    if args.shm:
        _init_shm(args.shm)
    elif args.shm_auto:
        # Auto-detect: wait for shared memory segments to appear in /dev/shm
        print("[cmd] Waiting for shared memory segments...")
        for _ in range(120):
            shm_files = [f for f in os.listdir("/dev/shm") if f.startswith("psm_")]
            if shm_files:
                # The run_command output shm is typically the 2nd or 3rd created
                # Try each one — the right one will have run_command data
                for name in sorted(shm_files):
                    _init_shm(name)
                    if _shm_writer:
                        break
                if _shm_writer:
                    break
            time.sleep(5)
        if not _shm_writer:
            print("[cmd] No shared memory found after 10 minutes")

    threading.Thread(target=lambda: ThreadingHTTPServer(
        ("0.0.0.0", LISTEN_PORT), HlsHandler).serve_forever(),
        daemon=True).start()
    print(f"[hls] Server at http://localhost:{LISTEN_PORT}")

    threading.Thread(target=lambda: ffmpeg_writer(cpu=args.cpu), daemon=True).start()
    zmq_receiver(args.host, args.port)


if __name__ == "__main__":
    main()
