"""WBC handler for SONIC whole-body control (G1 humanoid).

Contains all SONIC-specific logic:
  - IsaacLab <-> MuJoCo joint remapping
  - Observation scaling (dof_pos, dof_vel, base_ang_vel)
  - Rolling history buffers (ang_vel, jpos, jvel, action, gravity)
  - Gravity orientation computation
  - Encoder mode, standing height signals
  - Action remapping with per-joint scaling

Usage in YAML:
    configuration:
      wbc:
        handler: "handlers.sonic_wbc.SonicWBCHandler"
"""

import numpy as np


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _quat_rotate_inverse(quat, v):
    w, x, y, z = quat
    qw, qx, qy, qz = w, -x, -y, -z
    return np.array([
        v[0]*(qw**2+qx**2-qy**2-qz**2) + v[1]*2*(qx*qy-qw*qz) + v[2]*2*(qx*qz+qw*qy),
        v[0]*2*(qx*qy+qw*qz) + v[1]*(qw**2-qx**2+qy**2-qz**2) + v[2]*2*(qy*qz-qw*qx),
        v[0]*2*(qx*qz-qw*qy) + v[1]*2*(qy*qz+qw*qx) + v[2]*(qw**2-qx**2-qy**2+qz**2),
    ], dtype=np.float32)


def get_gravity_orientation(quat):
    return _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))


# ---------------------------------------------------------------------------
# SONIC WBC Handler
# ---------------------------------------------------------------------------

class SonicWBCHandler:
    """WBC handler for SONIC encoder-decoder policy on Unitree G1.

    Reads all parameters from configuration.wbc in the robot YAML.
    """

    def setup(self, bus, config, state):
        """Initialize from YAML config, set initial qpos, return n_actuated.

        ``state`` is a RobotState whose qpos/qvel may alias the engine's
        internal arrays (zero-copy for MuJoCo).  The engine handles forward
        kinematics after this method returns.
        """
        wbc = config.get("configuration", {}).get("wbc", {})

        self._num_actions = int(wbc.get("num_actions", 29))
        self._default_angles = np.array(wbc.get("default_angles", [0.0] * self._num_actions), dtype=np.float32)
        self._kps = np.array(wbc.get("kps", [100.0] * self._num_actions), dtype=np.float32)
        self._kds = np.array(wbc.get("kds", [5.0] * self._num_actions), dtype=np.float32)
        self._action_scales = np.array(wbc.get("action_scales", [1.0] * self._num_actions), dtype=np.float32)
        self._encoder_mode = int(wbc.get("encoder_mode", 0))
        self._history_frames = int(wbc.get("history_frames", 10))

        self._mj2il = wbc.get("isaaclab_to_mujoco", list(range(self._num_actions)))
        self._il2mj = wbc.get("mujoco_to_isaaclab", list(range(self._num_actions)))

        n_joints = state.qpos.shape[0] - 7
        self._n_joints = n_joints
        self._n_actuated = min(n_joints, self._num_actions)

        for mj_i in range(self._n_actuated):
            state.qpos[7 + mj_i] = self._default_angles[mj_i]

        self._standing_height = state.qpos[2]

        self._target = self._default_angles.copy()

        hf = self._history_frames
        na = self._num_actions
        self._ang_vel_hist = np.zeros((hf, 3), dtype=np.float32)
        self._jpos_hist = np.zeros((hf, na), dtype=np.float32)
        self._jvel_hist = np.zeros((hf, na), dtype=np.float32)
        self._action_hist = np.zeros((hf, na), dtype=np.float32)
        self._gravity_hist = np.zeros((hf, 3), dtype=np.float32)
        self._last_action_il = np.zeros(na, dtype=np.float32)

        self._qpos_history = np.zeros((4, 36), dtype=np.float32)
        initial_qpos = state.qpos[:min(36, len(state.qpos))].astype(np.float32)
        for i in range(4):
            self._qpos_history[i, :len(initial_qpos)] = initial_qpos

        control_hz = float(config.get("control_frequency_hz", 50))
        planner_traj_hz = 30.0
        self._ref_frame = 0.0
        self._ref_frame_advance = planner_traj_hz / control_hz
        self._last_traj_id = None
        self._held_traj = None
        self._min_frames_before_replan = 18

        self._blend_ticks_total = 8
        self._blend_tick = self._blend_ticks_total
        self._blend_old_jpos = None
        self._blend_old_jvel = None
        self._blend_old_rz = None
        self._blend_old_orn = None

        bus.put("standing_height", np.array([self._standing_height], dtype=np.float32))
        bus.put("encoder_mode", np.array([self._encoder_mode], dtype=np.float32))
        bus.put("base_quaternion", state.qpos[3:7].copy().astype(np.float64))
        bus.put("context_mujoco_qpos", self._qpos_history.copy())

        return self._n_actuated

    def pre_step(self, bus, state):
        """Read robot state, compute derived quantities, update history, write to bus."""
        nj = self._n_joints
        na = self._num_actions
        qj = state.qpos[7:7 + nj].copy()
        dqj = state.qvel[6:6 + nj].copy()
        quat = state.qpos[3:7].copy()
        omega = state.qvel[3:6].copy()
        gravity = get_gravity_orientation(quat)

        qj_il = np.zeros(na, dtype=np.float32)
        dqj_il = np.zeros(na, dtype=np.float32)
        omega_il = omega.astype(np.float32)
        for mj_i in range(min(nj, na)):
            il_i = self._mj2il[mj_i]
            qj_il[il_i] = qj[mj_i] - self._default_angles[mj_i]
            dqj_il[il_i] = dqj[mj_i]

        self._ang_vel_hist = np.roll(self._ang_vel_hist, -1, axis=0)
        self._jpos_hist = np.roll(self._jpos_hist, -1, axis=0)
        self._jvel_hist = np.roll(self._jvel_hist, -1, axis=0)
        self._action_hist = np.roll(self._action_hist, -1, axis=0)
        self._gravity_hist = np.roll(self._gravity_hist, -1, axis=0)
        self._ang_vel_hist[-1] = omega_il
        self._jpos_hist[-1] = qj_il
        self._jvel_hist[-1] = dqj_il
        self._action_hist[-1] = self._last_action_il
        self._gravity_hist[-1] = gravity

        self._qpos_history = np.roll(self._qpos_history, -1, axis=0)
        frame = state.qpos[:min(36, len(state.qpos))].astype(np.float32)
        self._qpos_history[-1, :] = 0.0
        self._qpos_history[-1, :len(frame)] = frame
        bus.put("context_mujoco_qpos", self._qpos_history.copy())

        bus.put("base_quaternion", quat.astype(np.float64))
        bus.put("angular_velocity", omega.astype(np.float32))
        bus.put("joint_positions", qj.astype(np.float32))
        bus.put("joint_velocities", dqj.astype(np.float32))
        bus.put("standing_height", np.array([self._standing_height], dtype=np.float32))
        bus.put("encoder_mode", np.array([self._encoder_mode], dtype=np.float32))
        bus.put("gravity", gravity)

        bus.put("ang_vel_hist", self._ang_vel_hist.flatten())
        bus.put("jpos_hist", self._jpos_hist.flatten())
        bus.put("jvel_hist", self._jvel_hist.flatten())
        bus.put("action_hist", self._action_hist.flatten())
        bus.put("gravity_hist", self._gravity_hist.flatten())

        self._extract_reference_trajectory(bus)

    # Number of lower-body joints in IsaacLab order (6 left leg + 6 right leg)
    _N_LOWER_BODY = 12

    # Encoder needs 10 look-ahead frames at "step 5" (50 Hz) = step 3 at 30 Hz.
    _REF_STEP_30HZ = 3
    _REF_N_SAMPLES = 10

    def _extract_reference_trajectory(self, bus):
        """Read the planner's reference_trajectory from the bus, advance a
        frame cursor (matching the C++ ``current_frame_`` logic), and write
        per-frame reference signals that the encoder handler consumes.

        Frame advancement:
          The planner trajectory is at 30 Hz.  Each 50 Hz control tick we
          advance the cursor by 30/50 = 0.6 frames.  When a **new** planner
          trajectory arrives (detected via array identity), the cursor resets
          to 0 so the 10-frame observation window starts from the new plan's
          beginning (mirrors the C++ reset-on-blend behaviour).

        Writes (when trajectory available):
          ref_joint_positions_10f  : [10, 29] IsaacLab order
          ref_joint_velocities_10f : [10, 29] IsaacLab order
          ref_root_z_10f           : [10]
          ref_orientation_10f      : [10, 4]  quaternion (w,x,y,z)
          ref_root_z               : [1]      current frame root z
          ref_orientation          : [4]      current frame quat
        """
        bus_traj = bus.get("reference_trajectory")
        if bus_traj is None:
            return

        bus_traj_id = id(bus_traj)
        is_new_on_bus = bus_traj_id != self._last_traj_id

        if self._held_traj is None:
            self._held_traj = np.asarray(bus_traj, dtype=np.float32).copy()
            self._last_traj_id = bus_traj_id
            self._ref_frame = 0.0
        elif is_new_on_bus and self._ref_frame >= self._min_frames_before_replan:
            if self._blend_old_jpos is not None:
                self._blend_tick = 0
            self._held_traj = np.asarray(bus_traj, dtype=np.float32).copy()
            self._last_traj_id = bus_traj_id
            self._ref_frame = 0.0
        elif is_new_on_bus:
            self._last_traj_id = bus_traj_id
            self._ref_frame += self._ref_frame_advance
        else:
            self._ref_frame += self._ref_frame_advance

        traj = self._held_traj
        if traj.ndim == 3:
            traj = traj[0]  # [N, 36]
        n_frames = traj.shape[0]
        if n_frames < 1:
            return

        npf = bus.get("num_pred_frames")
        if npf is not None:
            n_valid = min(int(npf.flat[0]), n_frames)
        else:
            n_valid = n_frames

        base_frame = int(self._ref_frame)

        na = self._num_actions
        il2mj = self._il2mj
        step = self._REF_STEP_30HZ
        n_samples = self._REF_N_SAMPLES

        jpos_il = np.zeros((n_samples, na), dtype=np.float32)
        root_z = np.zeros(n_samples, dtype=np.float32)
        orientations = np.zeros((n_samples, 4), dtype=np.float64)

        for si in range(n_samples):
            fi = min(base_frame + si * step, n_valid - 1)
            frame = traj[fi]

            root_z[si] = frame[2]
            orientations[si] = frame[3:7]

            for il_i in range(na):
                mj_i = il2mj[il_i]
                jpos_il[si, il_i] = frame[7 + mj_i] if (7 + mj_i) < 36 else 0.0

        jvel_il = np.zeros_like(jpos_il)
        dt_30 = 1.0 / 30.0
        for si in range(n_samples - 1):
            jvel_il[si] = (jpos_il[si + 1] - jpos_il[si]) / (dt_30 * step)
        if n_samples > 1:
            jvel_il[-1] = jvel_il[-2]

        if self._blend_tick < self._blend_ticks_total:
            alpha = (self._blend_tick + 1) / self._blend_ticks_total
            jpos_il = alpha * jpos_il + (1.0 - alpha) * self._blend_old_jpos
            jvel_il = alpha * jvel_il + (1.0 - alpha) * self._blend_old_jvel
            root_z = alpha * root_z + (1.0 - alpha) * self._blend_old_rz
            orientations = alpha * orientations + (1.0 - alpha) * self._blend_old_orn
            self._blend_tick += 1

        self._blend_old_jpos = jpos_il.copy()
        self._blend_old_jvel = jvel_il.copy()
        self._blend_old_rz = root_z.copy()
        self._blend_old_orn = orientations.copy()

        bus.put("ref_joint_positions_10f", jpos_il)
        bus.put("ref_joint_velocities_10f", jvel_il)
        bus.put("ref_root_z_10f", root_z)
        bus.put("ref_orientation_10f", orientations.astype(np.float64))
        bus.put("ref_root_z", np.array([root_z[0]], dtype=np.float32))
        bus.put("ref_orientation", orientations[0].astype(np.float64))

    _diag_count = 0
    _diag_limit = 30
    _diag_walking = False
    _diag_has_ref = False

    def reset_diag(self):
        self._diag_count = 0
        self._diag_walking = False
        self._diag_has_ref = False

    def post_step(self, bus):
        """Read action from bus, remap IsaacLab->MuJoCo, return (target, kps, kds)."""
        action_raw = bus.get("action")
        if action_raw is not None:
            action_il = action_raw.flatten()
            self._last_action_il = action_il.copy()

            self._target = self._default_angles.copy()
            for mj_i in range(self._n_actuated):
                il_i = self._mj2il[mj_i]
                delta = action_il[il_i] * self._action_scales[mj_i]
                self._target[mj_i] = self._default_angles[mj_i] + delta

        mode = bus.get("mode")
        ref = bus.get("reference_trajectory")
        ref_jpos = bus.get("ref_joint_positions_10f")
        enc = bus.get("encoded_tokens")
        m = int(mode.flat[0]) if mode is not None else 0

        if m != 0 and not self._diag_walking:
            self._diag_walking = True
            print(f"\n[DIAG] Mode changed (mode={m}), waiting for ref_jpos...")

        if self._diag_walking and not self._diag_has_ref and ref_jpos is not None:
            self._diag_has_ref = True
            self._diag_count = 0
            print(f"[DIAG] === Walking ref active, capturing {self._diag_limit} ticks ===")

        if self._diag_walking and self._diag_has_ref and self._diag_count < self._diag_limit:
            self._diag_count += 1
            a_std = float(action_raw.std()) if action_raw is not None else 0
            a_max = float(np.abs(action_raw).max()) if action_raw is not None else 0
            t_delta = float(np.abs(self._target[:self._n_actuated] - self._default_angles[:self._n_actuated]).max())
            rf = self._ref_frame
            print(f"[DIAG] #{self._diag_count:2d} mode={m} "
                  f"ref_jpos={'yes' if ref_jpos is not None else 'NO'} "
                  f"enc={'yes' if enc is not None else 'NO'} "
                  f"rf={rf:.1f} "
                  f"a_std={a_std:.3f} a_max={a_max:.3f} "
                  f"tgt_delta={t_delta:.4f}")
            if self._diag_count == self._diag_limit:
                print("[DIAG] (suppressing further output)")

        return (self._target[:self._n_actuated],
                self._kps[:self._n_actuated],
                self._kds[:self._n_actuated])
