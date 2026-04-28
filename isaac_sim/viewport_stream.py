# This project was developed with assistance from AI tools.
"""HLS viewport streamer for Isaac Sim.

Captures frames via Replicator's RGB annotator, pipes raw RGBA to an
ffmpeg subprocess (NVENC H.264), and serves HLS segments over HTTP.
Open viewer.html in a browser to watch the simulation.

Adapted from industrial-ai-showcase viewport_mjpeg.py.
"""

import os
import queue
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
FFMPEG_PATH = "/tmp/ffmpeg"
HLS_DIR = "/tmp/hls"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8090

CAMERA_PRIM = "/RobotCamera"
RESOLUTION = (1280, 720)
CAPTURE_EVERY_N_TICKS = 2
TARGET_FPS = 30

_ffmpeg_proc = None
_ffmpeg_lock = threading.Lock()
_rgb_annotator = None
_update_sub = None
_tick = 0
_frame_count = 0
_setup_done = False
_frame_queue: queue.Queue = queue.Queue(maxsize=4)


def _download_ffmpeg() -> bool:
    if os.path.isfile(FFMPEG_PATH):
        return True
    print("[stream] downloading ffmpeg...", flush=True)
    try:
        import lzma
        import tarfile
        import urllib.request
        resp = urllib.request.urlopen(FFMPEG_URL, timeout=120)
        with lzma.open(resp) as xz:
            with tarfile.open(fileobj=xz, mode="r|") as tar:
                for m in tar:
                    if m.name.endswith("/bin/ffmpeg") and m.isfile():
                        f = tar.extractfile(m)
                        if f:
                            with open(FFMPEG_PATH, "wb") as out:
                                out.write(f.read())
                            os.chmod(FFMPEG_PATH, 0o755)
                            print("[stream] ffmpeg ready", flush=True)
                            return True
    except Exception:
        print(f"[stream] ffmpeg download failed: {traceback.format_exc()}", flush=True)
    return False


def _start_ffmpeg() -> None:
    global _ffmpeg_proc
    os.makedirs(HLS_DIR, exist_ok=True)
    for f in os.listdir(HLS_DIR):
        os.remove(os.path.join(HLS_DIR, f))
    w, h = RESOLUTION
    cmd = [
        FFMPEG_PATH, "-y",
        "-use_wallclock_as_timestamps", "1",
        "-f", "rawvideo",
        "-pixel_format", "rgba",
        "-video_size", f"{w}x{h}",
        "-framerate", str(TARGET_FPS),
        "-i", "pipe:0",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-tune", "ll",
        "-b:v", "4M",
        "-maxrate", "6M",
        "-bufsize", "8M",
        "-g", str(TARGET_FPS),
        "-keyint_min", str(TARGET_FPS),
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "8",
        "-hls_flags", "delete_segments+append_list+omit_endlist",
        "-hls_segment_type", "mpegts",
        os.path.join(HLS_DIR, "stream.m3u8"),
    ]
    _ffmpeg_proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    print(f"[stream] ffmpeg started ({w}x{h} -> HLS/NVENC)", flush=True)
    threading.Thread(target=_ffmpeg_stderr_reader, daemon=True).start()


def _ffmpeg_stderr_reader() -> None:
    proc = _ffmpeg_proc
    if proc is None or proc.stderr is None:
        return
    for line in proc.stderr:
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"[stream] ffmpeg: {text}", flush=True)


def _find_camera_prim() -> str:
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(CAMERA_PRIM).IsValid():
            return CAMERA_PRIM
        from pxr import UsdGeom
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                return str(prim.GetPath())
    except Exception:
        pass
    return CAMERA_PRIM


def _setup_render_product() -> bool:
    global _rgb_annotator, _setup_done

    try:
        import omni.usd
        ctx = omni.usd.get_context()
        if ctx is None or ctx.get_stage() is None:
            return False
    except Exception:
        return False

    try:
        import omni.replicator.core as rep
    except Exception:
        print(f"[stream] omni.replicator.core not available", flush=True)
        return False

    try:
        camera = _find_camera_prim()
        rp = rep.create.render_product(camera, resolution=RESOLUTION)
        _rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        _rgb_annotator.attach([rp.path])
        _setup_done = True
        print(f"[stream] render product on {camera} at {RESOLUTION}", flush=True)
        return True
    except Exception:
        print(f"[stream] render product setup failed: {traceback.format_exc()}", flush=True)
        return False


def _on_update(event) -> None:
    global _tick, _setup_done

    _tick += 1
    if _tick % CAPTURE_EVERY_N_TICKS != 0:
        return

    if not _setup_done:
        if _tick % (CAPTURE_EVERY_N_TICKS * 15) == 0:
            _setup_render_product()
        return

    if _rgb_annotator is None:
        return

    try:
        data = _rgb_annotator.get_data()
    except Exception:
        _setup_done = False
        return

    if data is None:
        return

    if isinstance(data, dict):
        data = data.get("data", None)
        if data is None:
            return

    if hasattr(data, "ndim"):
        if data.ndim != 3 or data.shape[0] == 0:
            return
        raw = data.tobytes()
    else:
        raw = bytes(data)

    if not raw:
        return

    try:
        _frame_queue.put_nowait(raw)
    except queue.Full:
        pass


def _writer_loop() -> None:
    global _ffmpeg_proc, _frame_count

    while True:
        raw = _frame_queue.get()

        if _ffmpeg_proc is None:
            if not os.path.isfile(FFMPEG_PATH):
                continue
            _start_ffmpeg()

        try:
            _ffmpeg_proc.stdin.write(raw)
            _ffmpeg_proc.stdin.flush()
            _frame_count += 1
            if _frame_count == 1:
                print(f"[stream] first frame written ({len(raw)} bytes)", flush=True)
            elif _frame_count % 300 == 0:
                print(f"[stream] {_frame_count} frames encoded", flush=True)
        except (BrokenPipeError, OSError):
            print("[stream] ffmpeg pipe broken, restarting", flush=True)
            with _ffmpeg_lock:
                try:
                    _ffmpeg_proc.kill()
                except Exception:
                    pass
                _ffmpeg_proc = None


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


class _HlsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
            return

        if self.path in ("/healthz", "/health"):
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if self.path.startswith("/hls/"):
            filename = self.path[5:].split("?")[0]
            if "/" in filename or ".." in filename:
                self.send_response(403)
                self.end_headers()
                return
            filepath = os.path.join(HLS_DIR, filename)
            if not os.path.isfile(filepath):
                self.send_response(404)
                self._cors()
                self.end_headers()
                return
            with open(filepath, "rb") as f:
                file_data = f.read()
            self.send_response(200)
            self._cors()
            if filename.endswith(".m3u8"):
                self.send_header("Content-Type", "application/vnd.apple.mpegurl")
            elif filename.endswith(".ts"):
                self.send_header("Content-Type", "video/mp2t")
            else:
                self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(file_data)))
            self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()
            try:
                self.wfile.write(file_data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_response(404)
        self._cors()
        self.end_headers()


def start() -> None:
    global _update_sub

    threading.Thread(target=_download_ffmpeg, daemon=True).start()
    threading.Thread(target=lambda: ThreadingHTTPServer(
        (LISTEN_HOST, LISTEN_PORT), _HlsHandler
    ).serve_forever(), daemon=True).start()
    threading.Thread(target=_writer_loop, daemon=True).start()

    try:
        import omni.kit.app
        _update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(_on_update, name="viewport_stream")
        )
        print(f"[stream] HLS server at http://localhost:{LISTEN_PORT}/hls/stream.m3u8", flush=True)
    except Exception:
        print(f"[stream] failed to subscribe to update stream: {traceback.format_exc()}", flush=True)
