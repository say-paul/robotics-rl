#!/usr/bin/env python
"""
G1 Orchestrator
===============
Entry-point that wires the physics simulation, WBC controller,
prefrontal cortex, and chatbot server into a running loop.

Usage
-----
    # Interactive with chatbot (accessible from any host)
    python orchestrator.py --scene flat_ground --headless --chatbot-port 9000

    # Run mission with recording
    python orchestrator.py --scene water_bottle_stage \\
        --mission missions/water_bottle_presentation_final.yaml \\
        --headless --record --auto-start

    # With VLM planner for intelligent command routing
    python orchestrator.py --scene water_bottle_stage --headless \\
        --vlm-model moondream --chatbot-port 9000
"""

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Force unbuffered output so logs appear immediately (critical for headless)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from cortex.peripheral.device_manager import detect_best_device

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from cortex.motor.controller import MotorCortex
from cortex.perefrontal.brain import PrefrontalCortex, ActionRequest, Intent
from scene.simulation import G1Simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
    stream=sys.stderr,
)
logger = logging.getLogger("orchestrator")


class Orchestrator:
    """Runs the simulation and coordinates all subsystems."""

    def __init__(
        self,
        scene_name: str = "flat_ground",
        headless: bool = True,
        record: bool = False,
        mission_file: Optional[str] = None,
        auto_start_mission: bool = False,
        controller_type: str = "groot",
        chatbot_port: int = 9000,
        vla_model: Optional[str] = None,
        vla_model_path: Optional[str] = None,
        vlm_model: Optional[str] = None,
        vlm_model_path: Optional[str] = None,
        planner_mode: str = "auto",
    ):
        self.headless = headless
        self.record = record
        self.chatbot_port = chatbot_port
        self.scene_name = scene_name
        self._planner_mode = planner_mode
        self._running = False
        self._stop_requested = False
        self._harness_releasing = False
        self._harness_lower_remaining = 0
        self._randomize_pending = False

        # Simulation (full physics)
        self.sim = G1Simulation(scene_name=scene_name, headless=headless)

        # Motor cortex (WBC controller)
        self.motor = MotorCortex(controller_type=controller_type)

        # Step odometry (foot-force step counting + IMU heading)
        self._odometry = None
        self._init_odometry()

        # Prefrontal cortex
        self.brain = PrefrontalCortex(motor=self.motor, auto_trigger=auto_start_mission)
        self.brain.set_speech_callback(self._on_speech)
        self.brain.scene_name = scene_name
        self.brain.set_robot_state_fn(
            lambda: (self.sim.get_base_position(), self.sim.get_base_yaw())
        )

        # Mission
        if mission_file:
            self._load_mission(mission_file)
            if auto_start_mission:
                self.brain.start_mission()

        # VLA model engine (pi0.5 — optional, for learned motor policies)
        self._vla_engine = None
        if vla_model:
            self._init_vla(vla_model, vla_model_path)

        # VLM engine + Planner
        # Planner is ALWAYS created for rule-based commands.
        # VLM is optional -- adds intelligent intent classification on top.
        self._vlm_engine = None
        self._planner = None
        self._scene_context = ""
        self._landmark_map = {}
        self._init_planner(vlm_model, vlm_model_path)

        # Recording
        self._video_frames: list = []

    def _init_odometry(self) -> None:
        """Initialize step odometry from MuJoCo sensor data."""
        try:
            from cortex.peripheral.odometry import StepOdometry
            self._odometry = StepOdometry(self.sim.model)
            logger.info("Step odometry initialized")
        except Exception as e:
            logger.warning("Step odometry unavailable: %s", e)

    def _init_vla(self, model_name: str, model_path: Optional[str]) -> None:
        from cortex.peripheral.vla_engine import VLAEngine
        from cortex.peripheral.action_mapper import ActionMapper, MappingMode

        dev = detect_best_device()
        logger.info("VLA: initializing %s on %s", model_name, dev.name)
        self._vla_engine = VLAEngine(
            model_name=model_name, model_path=model_path, compile_model=True,
        )
        mapper = ActionMapper(mode=MappingMode.VELOCITY, velocity_scale=0.5)
        self.brain.set_vla_engine(self._vla_engine, mapper)

    def _init_planner(self, vlm_model: Optional[str], vlm_model_path: Optional[str]) -> None:
        """Initialize scene context, tool registry, and planner.

        The planner is ALWAYS created for rule-based command handling.
        VLM is optionally layered on top for intelligent intent classification.
        """
        from cortex.peripheral.planner import Planner, ToolRegistry
        from cortex.peripheral.scene_context import (
            build_environment_context,
            get_landmark_map,
        )

        self._scene_context = build_environment_context(self.scene_name)
        self._landmark_map = get_landmark_map(self.scene_name)
        logger.info(
            "Scene context loaded (%d chars, %d landmarks)",
            len(self._scene_context), len(self._landmark_map),
        )

        if vlm_model:
            from cortex.peripheral.vlm_engine import VLMEngine
            self._vlm_engine = VLMEngine(
                model_name=vlm_model, model_path=vlm_model_path, compile_model=True,
            )

        # Wire motor cortex with sensors for the blocking movement API
        self.motor.wire(
            odometry=self._odometry,
            sim_state_fn=lambda: (self.sim.get_base_position(), self.sim.get_base_yaw()),
        )

        tools = ToolRegistry()
        tools.wire(
            motor=self.motor,
            get_image_fn=lambda: self.sim.get_cached_frame("head_camera"),
            vlm_caption_fn=lambda img, question="Describe what you see.": (
                self._vlm_engine.caption(img, question)
                if self._vlm_engine and self._vlm_engine.loaded else "VLM not loaded"
            ),
            scene_context=self._scene_context,
            start_mission_fn=lambda: self.brain.start_mission(),
            get_status_fn=lambda: self.brain.generate_status_text(),
            quit_simulation_fn=lambda: self._request_quit(),
            release_harness_fn=lambda: self._release_harness(),
        )

        pm = (self._planner_mode or "auto").lower().strip()
        if pm not in ("auto", "model", "embedding", "rules"):
            logger.warning("Unknown planner_mode %r — using auto", pm)
            pm = "auto"

        self._planner = Planner(
            tools=tools,
            generate_text_fn=(
                (lambda prompt, **kw: self._vlm_engine.generate_text(prompt, **kw))
                if self._vlm_engine else lambda prompt, **kw: ""
            ),
            planner_mode=pm,
        )

        tools.set_goto_landmark(
            lambda name: self._planner._navigate_to_landmark(
                name,
                self._landmark_map,
                tuple(self.sim.get_base_position()[:2]),
                float(self.sim.get_base_yaw()),
            ),
        )

        # Randomize blocks (thread-safe)
        tools.set_randomize_fn(self._randomize_blocks)

        # Visual navigation: provide sim state so goto_visual_target can track position
        tools.set_sim_state_fn(
            lambda: (self.sim.get_base_position(), self.sim.get_base_yaw()),
        )

        # Intent matcher (fallback when model plan fails or planner-mode embedding)
        from cortex.peripheral.intent_matcher import IntentMatcher
        self._intent_matcher = IntentMatcher()
        if self._intent_matcher.load():
            self._planner.set_intent_matcher(self._intent_matcher)

        self.brain.set_planner(self._planner)
        logger.info(
            "Planner initialized (mode=%s, VLM: %s, IntentMatcher: %s)",
            pm, vlm_model or "none", "ready" if self._intent_matcher.loaded else "failed",
        )

    def _load_mission(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.error("Mission file not found: %s", p)
            return
        with open(p) as f:
            mc = yaml.safe_load(f)
        self.brain.load_mission(mc)
        logger.info("Mission loaded from %s", p)

    def _release_harness(self) -> None:
        """Schedule a gradual harness release without blocking the sim loop."""
        if self._harness_releasing:
            return
        self._harness_releasing = True
        self._harness_lower_remaining = 10
        logger.info("Releasing harness: lowering...")

    def _randomize_blocks(self) -> str:
        """Thread-safe: set flag and wait for sim loop to do the actual mutation."""
        self._randomize_pending = True
        deadline = time.monotonic() + 5.0
        while self._randomize_pending and time.monotonic() < deadline:
            time.sleep(0.05)
        positions = self.sim.get_marker_positions()
        lines = ["Blocks randomized to new positions:"]
        for name, pos in sorted(positions.items()):
            lines.append(f"  {name}: ({pos[0]:.1f}, {pos[1]:.1f})")
        return "\n".join(lines)

    def _request_quit(self) -> None:
        """Called by the quit_simulation tool."""
        self._stop_requested = True

    # ── Server ──────────────────────────────────────────────────────────────
    def _start_server(self) -> None:
        from server.app import create_app, set_shared_state
        import uvicorn

        set_shared_state(self.brain, self.sim, self)
        app = create_app()

        server_cfg = uvicorn.Config(
            app, host="0.0.0.0", port=self.chatbot_port, log_level="warning",
            ws_ping_interval=None, ws_ping_timeout=None,
        )
        server = uvicorn.Server(server_cfg)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        logger.info("Chatbot server on http://0.0.0.0:%d  (accessible externally)", self.chatbot_port)

    # ── Main Loop ───────────────────────────────────────────────────────────
    def run(self, max_duration: float = 0) -> Optional[str]:
        """Run the simulation. Returns path to recorded video or None."""
        self._running = True
        self._stop_requested = False

        logger.info("Loading WBC controller...")
        self.motor.load_controller()

        if self.motor.controller_kp is not None:
            self.sim.set_pd_gains(self.motor.controller_kp, self.motor.controller_kd)

        # Start chatbot server FIRST so user can interact while models load
        self._start_server()
        logger.info("=== Chatbot ready — send 'release' to drop harness, 'help' for commands ===")

        # Load VLA model (if configured)
        if self._vla_engine is not None:
            logger.info("Loading VLA model...")
            if self._vla_engine.load():
                self._vla_engine.start_background(
                    get_image_fn=lambda: self.sim.get_cached_frame("head_camera"),
                    get_state_fn=lambda: np.array(
                        [self.sim.data.qpos[self.sim.model.joint(j).qposadr[0]]
                         for j in self.sim._body_jnt_ids],
                        dtype=np.float32,
                    ),
                )
                logger.info("VLA model running in background on %s", self._vla_engine.device_type.name)
            else:
                logger.warning("VLA model failed to load — using rule-based fallback")

        # Load VLM model in background so sim loop can start immediately
        if self._vlm_engine is not None:
            def _load_vlm():
                logger.info("Loading VLM model (background)...")
                if self._vlm_engine.load():
                    logger.info("VLM ready on %s", self._vlm_engine.device_type.name)
                    if self._planner:
                        self._planner.set_vlm_available(True)
                        logger.info("Planner switched to VLM-first mode")
                else:
                    logger.warning("VLM failed to load — planner will use rule-based fallback only")
            threading.Thread(target=_load_vlm, daemon=True).start()

        logger.info("=== Starting simulation ===")
        logger.info("Harness: chat 'release' to drop, or viewer keys 7/8/9. 'quit' to end.")

        self.motor.start_stabilising()
        wbc_dt = self.motor.control_dt
        substeps = max(1, int(wbc_dt / self.sim.model.opt.timestep))
        viewer_every = max(1, int(0.02 / self.sim.model.opt.timestep))
        record_every = max(1, int((1.0 / 30) / self.sim.model.opt.timestep))
        cam_cache_every = max(1, int(0.066 / self.sim.model.opt.timestep))
        wall_start = time.perf_counter()
        global_step = 0
        _infinite_linger_start: Optional[float] = None

        try:
            while self._running and self.sim.viewer_running and not self._stop_requested:
                t0 = time.perf_counter()

                request = self.brain.decide()
                if request is not None:
                    self._execute_request(request)

                if self._randomize_pending:
                    self.sim._randomize_markers()
                    self._randomize_pending = False

                if self._harness_releasing:
                    if self._harness_lower_remaining > 0:
                        self.sim.lower_harness(amount=0.05)
                        self._harness_lower_remaining -= 1
                    else:
                        self.sim.release_harness()
                        self.motor.start_walking()
                        self._harness_releasing = False
                        logger.info("Harness released, WBC walking mode active")

                ls = self.sim.get_lowstate()
                targets = self.motor.tick(ls)

                for _ in range(substeps):
                    if targets:
                        self.sim.step_with_targets(targets)
                    else:
                        self.sim.step_idle()
                    global_step += 1

                    # Tick odometry every physics step
                    if self._odometry:
                        self._odometry.tick(self.sim.data)

                    if global_step % viewer_every == 0:
                        self.sim.sync_viewer()

                    if self.record and global_step % record_every == 0:
                        self._video_frames.append(
                            self.sim.render_tracking(azimuth=150, elevation=-20, distance=3.0)
                        )

                    if global_step % cam_cache_every == 0:
                        self.sim.update_frame_cache()

                # Auto-stop for headless missions reaching infinite phase
                if self.headless and self.brain.mission_active:
                    cp = self.brain.current_phase
                    if cp and cp.get("infinite"):
                        if _infinite_linger_start is None:
                            _infinite_linger_start = time.perf_counter()
                        elif time.perf_counter() - _infinite_linger_start > 3.0:
                            logger.info("Infinite phase reached — stopping")
                            break

                if max_duration > 0 and (time.perf_counter() - wall_start) > max_duration:
                    logger.info("Max duration reached — stopping")
                    break

                if not self.headless:
                    elapsed = time.perf_counter() - t0
                    sleep_s = wbc_dt - elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self._running = False
            if self._vla_engine:
                self._vla_engine.close()
            if self._vlm_engine:
                self._vlm_engine.close()
            video_path = None
            if self.record and self._video_frames:
                video_path = self._save_recording()
            self.sim.close()
            logger.info("=== Orchestrator shut down ===")
            return video_path

    def _execute_request(self, req: ActionRequest) -> None:
        if req.intent == Intent.QUIT:
            self._stop_requested = True
        elif req.intent == Intent.STOP:
            self.motor.stop()
        elif req.intent == Intent.CHAT_COMMAND:
            if req.metadata.get("action") == "release_harness":
                pass  # harness controlled via MuJoCo viewer keys 7/8/9
        elif req.intent == Intent.PERCEPTION:
            self._handle_perception(req)

    def _handle_perception(self, req: ActionRequest) -> None:
        """Run the planner in a background thread so the sim loop continues."""
        instruction = req.metadata.get("instruction", "")
        if not instruction or not self._planner:
            return

        def _run():
            try:
                pos = self.sim.get_base_position()
                yaw = self.sim.get_base_yaw()
                result = self._planner.execute(
                    instruction=instruction,
                    scene_context=self._scene_context,
                    robot_pos=(pos[0], pos[1]),
                    robot_yaw_rad=yaw,
                    landmark_map=self._landmark_map,
                )
                self.brain.push_response(result)
            except Exception as e:
                logger.error("Planner error: %s", e, exc_info=True)
                self.brain.push_response(f"Error processing request: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def request_stop(self) -> Optional[str]:
        """Called by server when user sends 'quit'. Returns video path."""
        self._stop_requested = True
        time.sleep(1)
        if self._video_frames:
            return self._save_recording()
        return None

    def _on_speech(self, text: str) -> None:
        logger.info("[SPEECH] %s", text)

    # ── Recording ───────────────────────────────────────────────────────────
    def _save_recording(self) -> str:
        cfg.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = cfg.VIDEO_DIR / f"recording_{ts}.mp4"
        try:
            import imageio
            writer = imageio.get_writer(str(out_path), fps=30)
            for frame in self._video_frames:
                writer.append_data(frame)
            writer.close()
            logger.info("Video saved: %s  (%d frames)", out_path, len(self._video_frames))
            return str(out_path)
        except ImportError:
            npy_path = out_path.with_suffix(".npy")
            np.save(str(npy_path), np.array(self._video_frames))
            logger.info("Frames saved as %s", npy_path)
            return str(npy_path)


def main():
    parser = argparse.ArgumentParser(description="G1 Deployment Orchestrator")
    parser.add_argument("--scene", default="flat_ground", choices=G1Simulation.list_scenes())
    parser.add_argument("--mission", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--controller", default="groot", choices=["groot", "holosoma"])
    parser.add_argument("--chatbot-port", type=int, default=9000, help="Chatbot server port")
    parser.add_argument("--max-duration", type=float, default=0)
    parser.add_argument("--vla-model", type=str, default=None, choices=["pi05", "openvla"],
                        help="VLA model to load (pi05 or openvla)")
    parser.add_argument("--vla-model-path", type=str, default=None,
                        help="Path to VLA model weights (default: models/<model_name>)")
    parser.add_argument("--vlm-model", type=str, default=None,
                        choices=["paligemma", "paligemma2", "paligemma3b", "qwen2vl", "moondream",
                                 "smolvlm", "smolvlm500m", "gemma3"],
                        help="VLM model for perception + planning")
    parser.add_argument("--vlm-model-path", type=str, default=None,
                        help="Path to VLM model weights")
    parser.add_argument(
        "--planner-mode",
        type=str,
        default="auto",
        choices=["auto", "model", "embedding", "rules"],
        help=(
            "auto: JSON task chain via VLM when loaded, else embedding/rules. "
            "model: VLM plan only (no regex/embedding). embedding: sentence-embeddings. "
            "rules: regex fallback only."
        ),
    )
    args = parser.parse_args()

    orch = Orchestrator(
        scene_name=args.scene,
        headless=args.headless,
        record=args.record,
        mission_file=args.mission,
        auto_start_mission=args.auto_start,
        controller_type=args.controller,
        chatbot_port=args.chatbot_port,
        vla_model=args.vla_model,
        vla_model_path=args.vla_model_path,
        vlm_model=args.vlm_model,
        vlm_model_path=args.vlm_model_path,
        planner_mode=args.planner_mode,
    )
    video = orch.run(max_duration=args.max_duration)
    if video:
        print(f"\nRecording saved: {video}")


if __name__ == "__main__":
    main()
