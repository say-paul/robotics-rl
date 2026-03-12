"""
Video streaming endpoints: WebSocket (primary) and MJPEG (fallback).

Both consume JPEG frames from the shared FrameBuffer.
"""

from flask import Response


def register_stream(app, sock, ctx, frames):
    """Attach WebSocket and MJPEG streaming routes to *app*."""

    @sock.route("/ws/stream")
    def ws_stream(ws):
        """Push binary JPEG frames over WebSocket as fast as the renderer produces."""
        last_seq = -1
        while ctx.sim_running:
            buf, seq = frames.get(last_seq, timeout=0.2)
            if not ctx.sim_running:
                break
            if buf is None or seq == last_seq:
                continue
            frame = buf
            last_seq = seq
            try:
                ws.send(frame)
            except Exception:
                break

    def _generate_mjpeg():
        last_seq = -1
        while ctx.sim_running:
            buf, seq = frames.get(last_seq, timeout=0.1)
            if not ctx.sim_running:
                break
            if buf is None or seq == last_seq:
                continue
            frame = buf
            last_seq = seq
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                b"\r\n" + frame + b"\r\n"
            )

    @app.route("/stream")
    def stream():
        return Response(
            _generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no",
            },
            direct_passthrough=True,
        )
