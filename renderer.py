"""
Offscreen EGL renderer and thread-safe frame buffer.

FrameBuffer is the clean seam for future remote rendering:
  - Local mode:  run_renderer() calls frames.put(jpeg_bytes) in-process.
  - Remote mode: swap run_renderer() for a network receiver that feeds
    frames.put(), or stream FrameBuffer contents to a remote web server.
"""

import time
import threading

import mujoco
import numpy as np
from turbojpeg import TurboJPEG

import config


class FrameBuffer:
    """Thread-safe JPEG frame exchange between a producer and N consumers."""

    def __init__(self):
        self._buf = None
        self._seq = 0
        self._cond = threading.Condition()

    @property
    def seq(self):
        return self._seq

    def put(self, jpeg_bytes):
        """Publish a new frame (called by the renderer thread)."""
        with self._cond:
            self._buf = jpeg_bytes
            self._seq += 1
            self._cond.notify_all()

    def get(self, last_seq, timeout=0.2):
        """
        Block until a frame newer than *last_seq* is available.

        Returns (jpeg_bytes | None, current_seq).
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq != last_seq, timeout=timeout
            )
            return self._buf, self._seq


_jpeg = TurboJPEG()


def run_renderer(ctx, frames, width, height):
    """
    EGL offscreen rendering loop.

    Reads MuJoCo scene state (under physics_lock), renders to RGB,
    encodes to JPEG via TurboJPEG, and publishes into *frames*.
    """
    model = ctx.mj_model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)

    out_buf = np.empty((height, width, 3), dtype=np.uint8)
    target_fps = int(1 / config.VIEWER_DT)
    print(f"[render] Offscreen renderer  {width}x{height} @ ~{target_fps} fps  (turbojpeg + websocket)")

    while ctx.sim_running:
        t0 = time.perf_counter()

        with ctx.physics_lock:
            renderer.update_scene(ctx.mj_data, ctx.cam)
        renderer.render(out=out_buf)

        jpeg_bytes = _jpeg.encode(out_buf, quality=80)
        frames.put(jpeg_bytes)

        elapsed = time.perf_counter() - t0
        sleep = config.VIEWER_DT - elapsed
        if sleep > 0:
            time.sleep(sleep)
