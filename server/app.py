"""
Web Server — externally accessible chatbot + camera stream + REST API
=====================================================================
Binds to 0.0.0.0 so it's reachable from any host on the network.
Port is configurable (default 9000).

Key features:
  - WebSocket chatbot for high-level commands
  - ALL commands routed through the VLM/Planner (no hardcoded templates)
  - "quit" keyword → stops simulation and returns video download link
  - REST endpoints for status, scenes, commands
  - Camera stream via WebSocket
"""

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

# Planner runs in a background thread (VLM + long WBC); 30s was far too short.
_CHAT_COMMAND_TIMEOUT_S = 7200.0

_brain = None
_sim = None
_orchestrator = None


def set_shared_state(brain, sim, orchestrator):
    global _brain, _sim, _orchestrator
    _brain = brain
    _sim = sim
    _orchestrator = orchestrator


def create_app() -> FastAPI:
    app = FastAPI(title="G1 Robot Control", docs_url="/docs")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _CHAT_HTML

    # ── Chat WebSocket ──────────────────────────────────────────────────────
    @app.websocket("/ws/chat")
    async def chat_ws(ws: WebSocket):
        await ws.accept()
        caps = []
        if _brain and hasattr(_brain, '_vla_engine') and _brain._vla_engine and _brain._vla_engine.loaded:
            caps.append(f"VLA: {_brain._vla_engine.model_name}")
        if _brain and hasattr(_brain, '_planner') and _brain._planner:
            caps.append("Planner: active")
        cap_str = f" [{', '.join(caps)}]" if caps else ""
        await ws.send_json({"role": "system", "text": (
            f"Connected to G1 robot.{cap_str}\n"
            "All commands are interpreted by the VLM planner.\n"
            "Try: 'walk 5 meters', 'go 10m east', 'look around', 'stop', 'help'\n"
            "Type 'quit' to end simulation and save video."
        )})
        try:
            while True:
                data = await ws.receive_text()
                cmd = data.strip()
                logger.info("[CMD:RECV] msg='%s' source=websocket", cmd)

                # Only "quit" is handled at the server level (safety escape)
                if cmd.lower() in ("quit", "exit", "end simulation", "shutdown"):
                    await ws.send_json({
                        "role": "system",
                        "text": "Stopping simulation... saving video...",
                        "request_done": False,
                    })
                    if _orchestrator is not None:
                        video_path = _orchestrator.request_stop()
                        if video_path:
                            await ws.send_json({
                                "role": "system",
                                "text": f"Video saved. Download: /api/video?path={video_path}",
                                "video_url": f"/api/video?path={video_path}",
                                "request_done": True,
                            })
                        else:
                            await ws.send_json({
                                "role": "system",
                                "text": "Simulation stopped (no recording).",
                                "request_done": True,
                            })
                    else:
                        await ws.send_json({"role": "system", "text": "Stopped.", "request_done": True})
                    break

                # Everything else goes to the brain → planner (wait until response is pushed)
                if _brain is not None:
                    _brain.enqueue_command(cmd)
                    deadline = time.monotonic() + _CHAT_COMMAND_TIMEOUT_S
                    response = None
                    while time.monotonic() < deadline:
                        await asyncio.sleep(0.15)
                        response = _brain.get_response(timeout=0)
                        if response is not None:
                            break
                    if response is not None:
                        await ws.send_json({"role": "bot", "text": response, "request_done": True})
                    else:
                        await ws.send_json({
                            "role": "error",
                            "text": "Timed out waiting for the robot to finish that command.",
                            "request_done": True,
                        })
                else:
                    await ws.send_json({
                        "role": "error",
                        "text": "Brain not initialized",
                        "request_done": True,
                    })

        except WebSocketDisconnect:
            logger.info("Chat client disconnected")

    # ── Camera WebSockets ───────────────────────────────────────────────────
    async def _stream_camera(ws: WebSocket, camera_name: str):
        await ws.accept()
        try:
            from PIL import Image
            while ws.client_state == WebSocketState.CONNECTED:
                if _sim is not None:
                    frame = _sim.get_cached_frame(camera_name)
                    if frame is not None:
                        try:
                            img = Image.fromarray(frame)
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=70)
                            await ws.send_bytes(buf.getvalue())
                        except Exception:
                            pass
                await asyncio.sleep(0.066)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/camera")
    async def camera_ws(ws: WebSocket):
        await _stream_camera(ws, "global_view")

    @app.websocket("/ws/camera/head")
    async def head_camera_ws(ws: WebSocket):
        await _stream_camera(ws, "head_camera")

    # ── REST Endpoints ──────────────────────────────────────────────────────
    @app.get("/api/status")
    async def status():
        if _brain is None or _sim is None:
            return {"status": "not_initialized"}
        motor = _brain.motor
        pos = _sim.get_base_position().tolist()
        return {
            "mode": motor.mode.name,
            "velocity": {
                "vx": motor.velocity_cmd.vx,
                "vy": motor.velocity_cmd.vy,
                "yaw": motor.velocity_cmd.yaw_rate,
            },
            "harness": _sim.harness_enabled,
            "base_position": pos,
            "base_height": pos[2],
            "mission_active": _brain.mission_active,
            "mission_progress": _brain.mission_progress,
            "phase": _brain.current_phase if _brain.current_phase else None,
            "nav_active": _brain.nav_active,
            "planner_active": _brain._planner is not None,
        }

    @app.post("/api/command")
    async def command(body: dict):
        cmd = body.get("command", "")
        if not cmd:
            return JSONResponse({"error": "no command"}, status_code=400)
        logger.info("[CMD:RECV] msg='%s' source=rest_api", cmd)

        if cmd.lower() in ("quit", "exit", "end simulation") and _orchestrator is not None:
            video = _orchestrator.request_stop()
            return {"status": "stopped", "video": video}

        if _brain is not None:
            _brain.enqueue_command(cmd)
            return {"status": "accepted", "command": cmd}
        return JSONResponse({"error": "not_initialized"}, status_code=503)

    @app.get("/api/scenes")
    async def scenes():
        from scene.simulation import G1Simulation
        return {"scenes": G1Simulation.list_scenes()}

    @app.get("/api/video")
    async def download_video(path: str):
        p = Path(path)
        if p.exists() and p.suffix in (".mp4", ".npy"):
            return FileResponse(str(p), filename=p.name, media_type="video/mp4")
        return JSONResponse({"error": "not found"}, status_code=404)

    return app


# ── Chat HTML Interface ────────────────────────────────────────────────────

_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 Robot Control</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;height:100vh;display:flex;flex-direction:column}
  header{background:#161b22;padding:12px 24px;border-bottom:1px solid #30363d;display:flex;align-items:center;gap:12px}
  header h1{font-size:1.2em;font-weight:600}
  .status-dot{width:10px;height:10px;border-radius:50%;background:#3fb950;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
  .main{flex:1;display:flex;overflow:hidden}
  .cam-panel{width:50%;display:flex;flex-direction:column;border-right:1px solid #30363d;min-width:0}
  .cam-box{flex:1;display:flex;flex-direction:column;border-bottom:1px solid #30363d;overflow:hidden;position:relative}
  .cam-box .label{position:absolute;top:6px;left:8px;background:rgba(0,0,0,0.65);padding:2px 8px;border-radius:4px;font-size:0.75em;color:#8b949e;z-index:1}
  .cam-box img{width:100%;height:100%;object-fit:contain;background:#000}
  .chat-panel{width:50%;display:flex;flex-direction:column;min-width:0}
  #messages{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
  .msg{padding:10px 14px;border-radius:8px;max-width:90%;font-size:0.9em;line-height:1.4;word-break:break-word;white-space:pre-wrap}
  .msg.system{background:#1f2937;color:#8b949e;align-self:center;text-align:center;font-size:0.8em;max-width:100%}
  .msg.user{background:#1f6feb;color:#fff;align-self:flex-end;border-bottom-right-radius:2px}
  .msg.bot{background:#21262d;color:#c9d1d9;align-self:flex-start;border-bottom-left-radius:2px}
  .msg.error{background:#3d1419;color:#f85149;align-self:center}
  .controls{border-top:1px solid #30363d;padding:10px 12px;display:flex;flex-direction:column;gap:6px}
  .quick-cmds{display:flex;gap:5px;flex-wrap:wrap}
  .quick-cmds button{padding:5px 10px;border:1px solid #30363d;border-radius:6px;background:#161b22;color:#8b949e;cursor:pointer;font-size:0.78em}
  .quick-cmds button:hover{background:#21262d;color:#c9d1d9}
  .quick-cmds button.danger{border-color:#da3633;color:#da3633}
  .quick-cmds button.danger:hover{background:#3d1419}
  .quick-cmds button.warn{border-color:#d29922;color:#d29922}
  .quick-cmds button.warn:hover{background:#3b2e00}
  .quick-cmds button.always-on{opacity:1!important;cursor:pointer!important}
  .input-bar{display:flex;gap:6px}
  .input-bar input{flex:1;padding:10px 14px;border:1px solid #30363d;border-radius:8px;background:#0d1117;color:#c9d1d9;font-size:0.95em;outline:none}
  .input-bar input:focus{border-color:#1f6feb}
  .input-bar input:disabled,.input-bar button:disabled{opacity:0.45;cursor:not-allowed}
  .quick-cmds button:disabled{opacity:0.45;cursor:not-allowed}
  .input-bar button{padding:10px 20px;border:none;border-radius:8px;background:#238636;color:#fff;font-weight:600;cursor:pointer;font-size:0.95em}
  .input-bar button:hover:not(:disabled){background:#2ea043}
  #video-link{display:none;padding:6px 14px;background:#1f6feb;color:#fff;text-decoration:none;border-radius:6px;text-align:center;font-size:0.85em}
  @media(max-width:768px){.main{flex-direction:column}.cam-panel,.chat-panel{width:100%}.cam-panel{max-height:40vh}}
</style>
</head>
<body>
<header>
  <div class="status-dot"></div>
  <h1>G1 Robot Control</h1>
</header>
<div class="main">
  <div class="cam-panel">
    <div class="cam-box">
      <span class="label">Head Camera (robot POV)</span>
      <img id="cam-head" alt="Head camera">
    </div>
    <div class="cam-box">
      <span class="label">Global View</span>
      <img id="cam-global" alt="Global view">
    </div>
  </div>
  <div class="chat-panel">
    <div id="messages"></div>
    <a id="video-link" href="#" target="_blank">Download Video</a>
    <div class="controls">
      <div class="quick-cmds">
        <button onclick="sendImmediate('release')" class="always-on">Release</button>
        <button onclick="sendImmediate('halt')" class="always-on warn">Halt</button>
        <button onclick="sendImmediate('stop')" class="always-on danger">Stop</button>
      </div>
      <div class="input-bar">
        <input id="inp" placeholder="Try: walk 5 meters, go 10m east, look around, stop, help..." autofocus
               onkeydown="if(event.key==='Enter'&&!chatBusy)send()">
        <button id="send-btn" type="button" onclick="send()">Send</button>
      </div>
    </div>
  </div>
</div>
<script>
const msgBox=document.getElementById('messages');
const inp=document.getElementById('inp');
const videoLink=document.getElementById('video-link');
const camHead=document.getElementById('cam-head');
const camGlobal=document.getElementById('cam-global');

let ws;
let chatBusy=false;
const sendBtn=document.getElementById('send-btn');
function setChatBusy(b){
  chatBusy=b;
  inp.disabled=b;
  if(sendBtn)sendBtn.disabled=b;
  document.querySelectorAll('.quick-cmds button').forEach(x=>{
    if(!x.classList.contains('always-on'))x.disabled=b;
  });
}
function sendImmediate(text){
  addMsg('user',text);
  if(ws&&ws.readyState===1)ws.send(text);
}
function connect(){
  const proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws/chat`);
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    addMsg(d.role||'bot',d.text||'');
    if(d.video_url){videoLink.href=d.video_url;videoLink.style.display='block'}
    if(Object.prototype.hasOwnProperty.call(d,'request_done')&&d.request_done)setChatBusy(false);
  };
  ws.onclose=()=>{setChatBusy(false);addMsg('system','Disconnected.');setTimeout(connect,3000)};
  ws.onerror=()=>ws.close();
}
function send(text){
  if(chatBusy)return;
  const t=text||inp.value.trim();if(!t)return;
  setChatBusy(true);
  addMsg('user',t);
  if(ws&&ws.readyState===1)ws.send(t);
  else setChatBusy(false);
  inp.value='';
}
function addMsg(role,text){
  const d=document.createElement('div');d.className='msg '+role;d.textContent=text;
  msgBox.appendChild(d);msgBox.scrollTop=msgBox.scrollHeight;
}

function connectCam(path,imgEl){
  const proto=location.protocol==='https:'?'wss':'ws';
  const cam=new WebSocket(`${proto}://${location.host}${path}`);
  cam.binaryType='blob';
  cam.onmessage=e=>{
    const url=URL.createObjectURL(e.data);
    imgEl.onload=()=>URL.revokeObjectURL(url);
    imgEl.src=url;
  };
  cam.onclose=()=>setTimeout(()=>connectCam(path,imgEl),2000);
  cam.onerror=()=>cam.close();
}

connect();
connectCam('/ws/camera/head',camHead);
connectCam('/ws/camera',camGlobal);
</script>
</body>
</html>"""
