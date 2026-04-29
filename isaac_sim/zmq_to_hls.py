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
import os
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import zmq

HLS_DIR = "/tmp/hls"
LISTEN_PORT = 8090
TARGET_FPS = 30

_frame_queue = queue.Queue(maxsize=4)

_INDEX_HTML = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Isaac Sim</title>
<style>*{margin:0;padding:0}body{background:#1e1e1e;display:flex;align-items:center;justify-content:center;height:100vh}
video{max-width:100%;max-height:100vh;object-fit:contain}
#s{position:fixed;top:10px;left:10px;color:#888;font:14px monospace}</style></head>
<body><div id="s">Connecting...</div><video id="v" autoplay muted playsinline></video>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<script>
const u='/hls/stream.m3u8',v=document.getElementById('v'),s=document.getElementById('s');
if(Hls.isSupported()){const h=new Hls({liveSyncDurationCount:2,liveMaxLatencyDurationCount:5,
liveDurationInfinity:true,lowLatencyMode:false,enableWorker:true,backBufferLength:0});
h.loadSource(u);h.attachMedia(v);h.on(Hls.Events.MANIFEST_PARSED,()=>{s.textContent='Playing';v.play()});
h.on(Hls.Events.ERROR,(_,d)=>{s.textContent='Error: '+d.details;if(d.fatal){
if(d.type===Hls.ErrorTypes.NETWORK_ERROR)h.startLoad();
else if(d.type===Hls.ErrorTypes.MEDIA_ERROR)h.recoverMediaError()}})}
else if(v.canPlayType('application/vnd.apple.mpegurl')){v.src=u;
v.addEventListener('playing',()=>{s.textContent='Playing'})}
else{s.textContent='HLS not supported'}
</script></body></html>"""


class HlsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

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
    args = parser.parse_args()

    threading.Thread(target=lambda: ThreadingHTTPServer(
        ("0.0.0.0", LISTEN_PORT), HlsHandler).serve_forever(),
        daemon=True).start()
    print(f"[hls] Server at http://localhost:{LISTEN_PORT}")

    threading.Thread(target=lambda: ffmpeg_writer(cpu=args.cpu), daemon=True).start()
    zmq_receiver(args.host, args.port)


if __name__ == "__main__":
    main()
