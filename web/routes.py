"""
HTTP routes: UI page, camera, harness, policy actions, reset, and state polling.
"""

import math
from glob import glob
from pathlib import Path

import mujoco
import numpy as np
from flask import render_template, request, jsonify

from actions import ACTION_REGISTRY

import config


def register_routes(app, ctx):
    """Attach all non-streaming routes to *app*."""

    # ---- Index ----

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            domain_id=config.DOMAIN_ID,
            interface=config.INTERFACE,
        )

    # ---- Camera ----

    @app.route("/camera", methods=["POST"])
    def camera_control():
        data = request.get_json(silent=True) or {}
        action = data.get("action", "")
        cam = ctx.cam

        if action == "orbit":
            s = 0.3
            cam.azimuth += data.get("dx", 0) * s
            cam.elevation = float(
                np.clip(cam.elevation + data.get("dy", 0) * s, -89, 89)
            )
        elif action == "pan":
            s = 0.005 * cam.distance
            dx, dy = data.get("dx", 0) * s, data.get("dy", 0) * s
            az = math.radians(cam.azimuth)
            el = math.radians(cam.elevation)
            cam.lookat[0] -= dx * math.cos(az) + dy * math.sin(az) * math.sin(el)
            cam.lookat[1] -= dx * math.sin(az) - dy * math.cos(az) * math.sin(el)
            cam.lookat[2] += dy * math.cos(el)
        elif action == "zoom":
            cam.distance = max(0.5, cam.distance + data.get("delta", 0) * 0.01)
        elif action == "reset":
            cam.distance = 3.0
            cam.azimuth = -130.0
            cam.elevation = -20.0
            cam.lookat[:] = [0.0, 0.0, 0.75]
        return jsonify(ok=True)

    # ---- Harness ----

    @app.route("/harness", methods=["POST"])
    def harness_update():
        data = request.get_json(silent=True) or {}
        h = ctx.harness
        if "height" in data:
            h.set_height(data["height"])
        if "stiffness" in data:
            h.set_stiffness(data["stiffness"])
        if "damping" in data:
            h.set_damping(data["damping"])
        if "enabled" in data:
            h.set_enabled(data["enabled"])
        return jsonify(ok=True)

    @app.route("/harness/preset", methods=["POST"])
    def harness_preset():
        data = request.get_json(silent=True) or {}
        name = data.get("preset", "")
        dispatch = {
            "full": ctx.harness.preset_full_support,
            "partial": ctx.harness.preset_partial_support,
            "light": ctx.harness.preset_light_support,
            "released": ctx.harness.preset_released,
        }
        fn = dispatch.get(name)
        if fn:
            fn()
        return jsonify(ok=True, preset=name)

    @app.route("/harness/auto_lower", methods=["POST"])
    def harness_auto_lower():
        data = request.get_json(silent=True) or {}
        ctx.harness.start_auto_lower(
            duration=data.get("duration", 10.0),
            target_height=data.get("target_height", 0.78),
            target_stiffness=data.get("target_stiffness", 0.0),
        )
        return jsonify(ok=True)

    @app.route("/harness/auto_stop", methods=["POST"])
    def harness_auto_stop():
        ctx.harness.stop_auto_lower()
        return jsonify(ok=True)

    # ---- Policy actions ----

    @app.route("/action/start", methods=["POST"])
    def action_start():
        data = request.get_json(silent=True) or {}
        name = data.get("name", "policy")

        cls = ACTION_REGISTRY.get(name)
        if cls is None:
            return jsonify(ok=False, error=f"Unknown action: {name}"), 400

        with ctx.action_lock:
            if ctx.active_action is not None and ctx.active_action.running:
                ctx.active_action.stop()

            kwargs = {}
            if name == "policy":
                kwargs["policy_path"] = data.get("policy_path", "")
                kwargs["device"] = data.get("device", "CPU")
                kwargs["vx"] = data.get("vx", 0.0)
                kwargs["vy"] = data.get("vy", 0.0)
                kwargs["vyaw"] = data.get("vyaw", 0.0)

            action = cls(**kwargs)
            ctx.dds_ready.wait(timeout=10.0)
            action.start(domain_id=None)
            ctx.active_action = action

        return jsonify(ok=True, action=name)

    @app.route("/action/stop", methods=["POST"])
    def action_stop():
        with ctx.action_lock:
            if ctx.active_action is not None:
                ctx.active_action.stop()
                ctx.active_action = None
        return jsonify(ok=True)

    # ---- Reset (respawn robot at default pose) ----

    @app.route("/reset", methods=["POST"])
    def reset_robot():
        with ctx.action_lock:
            if ctx.active_action is not None and ctx.active_action.running:
                ctx.active_action.stop()
                ctx.active_action = None

        ctx.harness.preset_full_support()

        with ctx.physics_lock:
            mujoco.mj_resetData(ctx.mj_model, ctx.mj_data)
            mujoco.mj_forward(ctx.mj_model, ctx.mj_data)

        return jsonify(ok=True)

    # ---- Available ONNX policies ----

    @app.route("/policies")
    def list_policies():
        onnx_files = sorted(
            p.name
            for p in Path("policies").glob("*.onnx")
            if not p.name.endswith(".onnx.data")
        )
        return jsonify(policies=onnx_files)

    # ---- Unified state ----

    @app.route("/state")
    def state():
        act_name = None
        with ctx.action_lock:
            if ctx.active_action is not None and ctx.active_action.running:
                act_name = ctx.active_action.__class__.__name__
        return jsonify(
            harness=ctx.harness.get_state(),
            camera=dict(
                azimuth=float(ctx.cam.azimuth),
                elevation=float(ctx.cam.elevation),
                distance=float(ctx.cam.distance),
            ),
            robot_z=float(ctx.robot_pos[2]),
            action=act_name,
        )
