"""
G1 Physics Simulation
=====================
MuJoCo-based simulation with:
  - Torque-controlled physics (not kinematics)
  - Elastic band (harness) for initial stabilisation
  - Direct WBC controller integration (Groot / Holosoma)
  - Offscreen rendering for headless recording

The robot is held by the elastic band at startup. Release it by calling
`release_harness()` (equivalent to pressing '9' in the LeRobot sim).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.spatial.transform

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from config import SCENE_DIR, NUM_JOINTS, SIM_DT

logger = logging.getLogger(__name__)

AVAILABLE_SCENES = {
    "flat_ground": "flat_ground.xml",
    "water_bottle_stage": "water_bottle_stage.xml",
}

# Default robot spawn for each scene (x, y, z)
SPAWN_POSITIONS = {
    "flat_ground": (0.0, 0.0, 0.75),
    "water_bottle_stage": (-5.0, 0.0, 0.75),
}

MARKER_NAMES = ["marker_1", "marker_2", "marker_3"]
MARKER_COLORS = {
    "marker_1": "red 1",
    "marker_2": "blue 2",
    "marker_3": "green 3",
}
_MARKER_Z = 0.914  # 3 feet above ground

# XY bounds for marker randomization per scene
_MARKER_BOUNDS = {
    "flat_ground":          {"x": (1.0, 5.0),  "y": (-3.0, 3.0)},
    "water_bottle_stage":   {"x": (-4.5, -2.0), "y": (-2.0, 2.0)},
}


# ── Elastic Band (harness) ─────────────────────────────────────────────────

class ElasticBand:
    """PD-controlled elastic band that holds the robot upright.
    Replicates the LeRobot unitree_sdk2py_bridge.ElasticBand."""

    def __init__(self):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        self.point = np.array([0.0, 0.0, 1.0])
        self.length = 0.0
        self.enable = True

    def advance(self, pose: np.ndarray) -> np.ndarray:
        """Compute 6D wrench [fx,fy,fz, tx,ty,tz] from current body pose.

        pose: 13D [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)]
        """
        pos = pose[0:3]
        quat_wxyz = pose[3:7]
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        dx = self.point - pos
        force = self.kp_pos * (dx + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)

        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = scipy.spatial.transform.Rotation.from_quat(quat_xyzw)
        rotvec = rot.as_rotvec()
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([force, torque])

    def raise_harness(self, amount: float = 0.1):
        self.length += amount

    def lower_harness(self, amount: float = 0.1):
        self.length -= amount

    def release(self):
        self.enable = not self.enable
        logger.info("Harness %s", "ENABLED" if self.enable else "RELEASED")


# ── Low State (mimics LeRobot G1_29_LowState) ──────────────────────────────

@dataclass
class _MotorState:
    q: float = 0.0
    dq: float = 0.0
    tau_est: float = 0.0
    temperature: float = 0.0

@dataclass
class _IMUState:
    quaternion: list = None
    gyroscope: list = None
    accelerometer: list = None
    rpy: list = None
    temperature: float = 0.0

    def __post_init__(self):
        self.quaternion = self.quaternion or [1.0, 0.0, 0.0, 0.0]
        self.gyroscope = self.gyroscope or [0.0, 0.0, 0.0]
        self.accelerometer = self.accelerometer or [0.0, 0.0, 0.0]
        self.rpy = self.rpy or [0.0, 0.0, 0.0]


class LowState:
    """Lightweight replacement for the DDS low-state message."""
    def __init__(self):
        self.motor_state = [_MotorState() for _ in range(NUM_JOINTS)]
        self.imu_state = _IMUState()
        self.mode_machine = 0


# ── Main Simulation ────────────────────────────────────────────────────────

class G1Simulation:
    """Full-physics MuJoCo simulation for the Unitree G1."""

    def __init__(
        self,
        scene_name: str = "flat_ground",
        headless: bool = True,
        render_size: Tuple[int, int] = (640, 480),
    ):
        if scene_name not in AVAILABLE_SCENES:
            raise ValueError(f"Unknown scene '{scene_name}'. Available: {list(AVAILABLE_SCENES)}")

        xml_path = SCENE_DIR / AVAILABLE_SCENES[scene_name]
        logger.info("Loading scene '%s' from %s", scene_name, xml_path)

        self.scene_name = scene_name
        self.headless = headless
        self.render_w, self.render_h = render_size

        # Load model
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        # Joint indices (body joints only — skip free-joint)
        self._body_jnt_ids = []
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if any(part in name for part in [
                "hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"
            ]):
                self._body_jnt_ids.append(i)
        self._body_jnt_ids = np.array(self._body_jnt_ids)
        assert len(self._body_jnt_ids) == NUM_JOINTS

        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

        # Elastic band (harness)
        self.band = ElasticBand()
        self.band_body_id = self.torso_body_id

        # Torque limits
        self.torque_limit = np.array([
            88, 88, 88, 139, 50, 50,    # left leg
            88, 88, 88, 139, 50, 50,    # right leg
            88, 50, 50,                  # waist
            25, 25, 25, 25, 25, 5, 5,   # left arm
            25, 25, 25, 25, 25, 5, 5,   # right arm
        ], dtype=np.float64)

        # PD gains — match LeRobot UnitreeG1Config defaults exactly
        self._kp = np.array([
            150, 150, 150, 300, 40, 40,   # left leg
            150, 150, 150, 300, 40, 40,   # right leg
            250, 250, 250,                 # waist
            50, 50, 80, 80, 40, 40, 40,   # left arm + wrist
            50, 50, 80, 80, 40, 40, 40,   # right arm + wrist
        ], dtype=np.float64)
        self._kd = np.array([
            2, 2, 2, 4, 2, 2,             # left leg
            2, 2, 2, 4, 2, 2,             # right leg
            5, 5, 5,                       # waist
            3, 3, 3, 3, 1.5, 1.5, 1.5,   # left arm + wrist
            3, 3, 3, 3, 1.5, 1.5, 1.5,   # right arm + wrist
        ], dtype=np.float64)
        self._active_kp = self._kp.copy()
        self._active_kd = self._kd.copy()

        # Viewer / renderer
        self._viewer = None
        self._renderer: Optional[mujoco.Renderer] = None

        # Frame cache for thread-safe access from the server
        import threading
        self._frame_lock = threading.Lock()
        self._cached_frames: Dict[str, np.ndarray] = {}
        self._marker_positions: Dict[str, np.ndarray] = {}

        if not headless and os.environ.get("DISPLAY"):
            try:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data,
                    key_callback=self._key_callback,
                    show_left_ui=False, show_right_ui=False,
                )
                self._viewer.cam.azimuth = 120
                self._viewer.cam.elevation = -30
                self._viewer.cam.distance = 6.0
                self._viewer.cam.lookat[:] = [-2, 0, 0.8]
            except Exception as exc:
                logger.warning("No viewer: %s", exc)
                self.headless = True
        else:
            self.headless = True

        # Reset to initial state
        self.reset()
        self._step_count = 0
        logger.info("G1Simulation ready  scene=%s  headless=%s", scene_name, self.headless)

    # ── Reset ───────────────────────────────────────────────────────────────
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # Set valid quaternion
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Spawn position
        spawn = SPAWN_POSITIONS.get(self.scene_name, (0, 0, 0.75))
        self.data.qpos[0] = spawn[0]
        self.data.qpos[1] = spawn[1]
        self.data.qpos[2] = spawn[2]

        # Update harness anchor to spawn point
        self.band.point = np.array([spawn[0], spawn[1], 1.0])
        self.band.enable = True
        self.band.length = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._randomize_markers()
        self._step_count = 0

    # ── Marker randomization ─────────────────────────────────────────────
    def _randomize_markers(self) -> None:
        """Place visual navigation markers at random floor positions."""
        bounds = _MARKER_BOUNDS.get(self.scene_name)
        if bounds is None:
            return
        xlo, xhi = bounds["x"]
        ylo, yhi = bounds["y"]
        for name in MARKER_NAMES:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name,
            )
            if body_id < 0:
                continue
            x = float(np.random.uniform(xlo, xhi))
            y = float(np.random.uniform(ylo, yhi))
            self.model.body_pos[body_id] = [x, y, _MARKER_Z]
            self._marker_positions[name] = np.array([x, y, _MARKER_Z])
        mujoco.mj_forward(self.model, self.data)
        logger.info(
            "Markers randomized: %s",
            {k: (f"{v[0]:.2f}", f"{v[1]:.2f}") for k, v in self._marker_positions.items()},
        )

    def get_marker_positions(self) -> Dict[str, np.ndarray]:
        """Ground-truth marker positions (debug / logging only)."""
        return dict(self._marker_positions)

    # ── Build Low State ─────────────────────────────────────────────────────
    def get_lowstate(self) -> LowState:
        """Build a LowState object from current MuJoCo data."""
        ls = LowState()
        for i, jnt_id in enumerate(self._body_jnt_ids):
            jnt = self.model.joint(jnt_id)
            qa = jnt.qposadr[0]
            da = jnt.dofadr[0]
            ls.motor_state[i].q = float(self.data.qpos[qa])
            ls.motor_state[i].dq = float(self.data.qvel[da])
            if i < self.model.nu:
                ls.motor_state[i].tau_est = float(self.data.actuator_force[i])

        ls.imu_state.quaternion = list(self.data.qpos[3:7])
        ls.imu_state.gyroscope = list(self.data.qvel[3:6])
        ls.imu_state.accelerometer = list(self.data.qacc[:3])
        return ls

    # ── Step with target positions ──────────────────────────────────────────
    def step_with_targets(self, target_qpos: Dict[str, float]) -> None:
        """Apply PD torque control toward target positions and step physics.

        target_qpos: dict mapping 'JointName.q' → float (radians)
                     (matches Groot/Holosoma output format)
        """
        from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

        cur_q = np.zeros(NUM_JOINTS)
        cur_dq = np.zeros(NUM_JOINTS)
        for i, jnt_id in enumerate(self._body_jnt_ids):
            jnt = self.model.joint(jnt_id)
            cur_q[i] = self.data.qpos[jnt.qposadr[0]]
            cur_dq[i] = self.data.qvel[jnt.dofadr[0]]

        targets = cur_q.copy()  # default: hold current position
        for joint in G1_29_JointIndex:
            key = f"{joint.name}.q"
            if key in target_qpos:
                targets[joint.value] = target_qpos[key]

        torques = self._active_kp * (targets - cur_q) + self._active_kd * (0 - cur_dq)
        torques = np.clip(torques, -self.torque_limit, self.torque_limit)

        # Apply elastic band
        if self.band.enable:
            pose = self._get_band_pose()
            wrench = self.band.advance(pose)
            self.data.xfrc_applied[self.band_body_id] = wrench
        else:
            self.data.xfrc_applied[self.band_body_id] = np.zeros(6)

        # Set control (torques on body joints)
        self.data.ctrl[:NUM_JOINTS] = torques

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

    def step_with_target_array(self, target_q: np.ndarray) -> None:
        """Apply PD torque control toward target joint positions and step physics.

        target_q: (29,) array of target joint angles in radians.
        No lerobot dependency — uses body_jnt_ids ordering directly.
        """
        cur_q = np.zeros(NUM_JOINTS)
        cur_dq = np.zeros(NUM_JOINTS)
        for i, jnt_id in enumerate(self._body_jnt_ids):
            jnt = self.model.joint(jnt_id)
            cur_q[i] = self.data.qpos[jnt.qposadr[0]]
            cur_dq[i] = self.data.qvel[jnt.dofadr[0]]

        torques = self._active_kp * (target_q - cur_q) + self._active_kd * (0 - cur_dq)
        torques = np.clip(torques, -self.torque_limit, self.torque_limit)

        if self.band.enable:
            pose = self._get_band_pose()
            self.data.xfrc_applied[self.band_body_id] = self.band.advance(pose)
        else:
            self.data.xfrc_applied[self.band_body_id] = np.zeros(6)

        self.data.ctrl[:NUM_JOINTS] = torques
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

    def step_idle(self) -> None:
        """Step physics with zero torque (just elastic band holds robot)."""
        if self.band.enable:
            pose = self._get_band_pose()
            self.data.xfrc_applied[self.band_body_id] = self.band.advance(pose)
        else:
            self.data.xfrc_applied[self.band_body_id] = np.zeros(6)

        self.data.ctrl[:] = 0
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

    def _get_band_pose(self) -> np.ndarray:
        """Get 13D pose for elastic band computation."""
        pos = self.data.xpos[self.band_body_id].copy()
        quat = self.data.xquat[self.band_body_id].copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.band_body_id,
            vel, 0,
        )
        lin_vel = vel[3:6]
        ang_vel = vel[0:3]
        return np.concatenate([pos, quat, lin_vel, ang_vel])

    def set_pd_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """Override PD gains (e.g., Holosoma provides its own via ONNX metadata)."""
        self._active_kp = np.array(kp, dtype=np.float64)
        self._active_kd = np.array(kd, dtype=np.float64)
        logger.info("PD gains updated from controller")

    # ── Harness controls ────────────────────────────────────────────────────
    def release_harness(self):
        self.band.release()

    def raise_harness(self, amount: float = 0.1):
        self.band.raise_harness(amount)

    def lower_harness(self, amount: float = 0.1):
        self.band.lower_harness(amount)

    @property
    def harness_enabled(self) -> bool:
        return self.band.enable

    # ── Rendering ───────────────────────────────────────────────────────────
    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.render_h, width=self.render_w,
            )
        return self._renderer

    def render_frame(self, camera: str = "global_view") -> np.ndarray:
        r = self._ensure_renderer()
        r.update_scene(self.data, camera=camera)
        return r.render()

    def render_tracking(
        self,
        azimuth: float = 150.0,
        elevation: float = -20.0,
        distance: float = 3.0,
    ) -> np.ndarray:
        """Render from a virtual camera that tracks the robot's base.

        azimuth:   horizontal angle in degrees (0 = +X, 90 = +Y, 150 ≈ 5 o'clock behind)
        elevation: vertical angle in degrees (negative = looking down)
        distance:  metres from the robot
        """
        r = self._ensure_renderer()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.get_base_position()
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        r.update_scene(self.data, camera=cam)
        return r.render()

    def update_frame_cache(self, cameras: list = None) -> None:
        """Render and cache frames for thread-safe access by the server.
        Must be called from the simulation thread."""
        if cameras is None:
            cameras = ["head_camera"]
        r = self._ensure_renderer()
        with self._frame_lock:
            for cam in cameras:
                try:
                    r.update_scene(self.data, camera=cam)
                    self._cached_frames[cam] = r.render().copy()
                except Exception:
                    pass
            # Tracking camera for the global view (5 o'clock, 20° elev, 3m)
            try:
                self._cached_frames["global_view"] = self.render_tracking(
                    azimuth=150, elevation=-20, distance=3.0,
                )
            except Exception:
                pass

    def get_cached_frame(self, camera: str = "global_view") -> Optional[np.ndarray]:
        """Thread-safe read of the last rendered frame (for server use)."""
        with self._frame_lock:
            return self._cached_frames.get(camera)

    def sync_viewer(self):
        if self._viewer is not None:
            self._viewer.sync()

    @property
    def viewer_running(self) -> bool:
        if self._viewer is None:
            return True
        return self._viewer.is_running()

    # ── Keyboard callback (for on-screen viewer) ───────────────────────────
    def _key_callback(self, key):
        import glfw
        if key == glfw.KEY_7:
            self.band.lower_harness()
            logger.info("Harness lowered (length=%.2f)", self.band.length)
        elif key == glfw.KEY_8:
            self.band.raise_harness()
            logger.info("Harness raised (length=%.2f)", self.band.length)
        elif key == glfw.KEY_9:
            self.band.release()

    # ── Queries ─────────────────────────────────────────────────────────────
    def get_base_position(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    def get_base_height(self) -> float:
        return float(self.data.qpos[2])

    def get_base_orientation(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def get_base_yaw(self) -> float:
        """Extract yaw (heading) angle in radians from the base quaternion."""
        quat_wxyz = self.data.qpos[3:7]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        rot = scipy.spatial.transform.Rotation.from_quat(quat_xyzw)
        return float(rot.as_euler("xyz")[2])

    # ── Cleanup ─────────────────────────────────────────────────────────────
    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @staticmethod
    def list_scenes():
        return list(AVAILABLE_SCENES.keys())
