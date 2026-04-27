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

    def setup(self, bus, config, model, data):
        """Initialize from YAML config, set initial qpos, return n_actuated."""
        wbc = config.get("configuration", {}).get("wbc", {})

        self._num_actions = int(wbc.get("num_actions", 29))
        self._default_angles = np.array(wbc.get("default_angles", [0.0] * self._num_actions), dtype=np.float32)
        self._kps = np.array(wbc.get("kps", [100.0] * self._num_actions), dtype=np.float32)
        self._kds = np.array(wbc.get("kds", [5.0] * self._num_actions), dtype=np.float32)
        self._action_scales = np.array(wbc.get("action_scales", [1.0] * self._num_actions), dtype=np.float32)
        self._obs_scales = wbc.get("obs_scales", {})
        self._encoder_mode = int(wbc.get("encoder_mode", 0))
        self._history_frames = int(wbc.get("history_frames", 10))

        self._mj2il = wbc.get("isaaclab_to_mujoco", list(range(self._num_actions)))
        self._il2mj = wbc.get("mujoco_to_isaaclab", list(range(self._num_actions)))

        n_joints = data.qpos.shape[0] - 7
        self._n_joints = n_joints
        self._n_actuated = min(n_joints, self._num_actions)

        for mj_i in range(self._n_actuated):
            data.qpos[7 + mj_i] = self._default_angles[mj_i]

        import mujoco as mj
        mj.mj_forward(model, data)
        self._standing_height = data.qpos[2]

        self._target = self._default_angles.copy()

        hf = self._history_frames
        na = self._num_actions
        self._ang_vel_hist = np.zeros((hf, 3), dtype=np.float32)
        self._jpos_hist = np.zeros((hf, na), dtype=np.float32)
        self._jvel_hist = np.zeros((hf, na), dtype=np.float32)
        self._action_hist = np.zeros((hf, na), dtype=np.float32)
        self._gravity_hist = np.zeros((hf, 3), dtype=np.float32)
        self._last_action_il = np.zeros(na, dtype=np.float32)

        bus.put("standing_height", np.array([self._standing_height], dtype=np.float32))
        bus.put("encoder_mode", np.array([self._encoder_mode], dtype=np.float32))
        bus.put("base_quaternion", data.qpos[3:7].copy().astype(np.float64))

        return self._n_actuated

    def pre_step(self, bus, model, data):
        """Read MuJoCo state, compute derived quantities, update history, write to bus."""
        nj = self._n_joints
        na = self._num_actions
        qj = data.qpos[7:7 + nj].copy()
        dqj = data.qvel[6:6 + nj].copy()
        quat = data.qpos[3:7].copy()
        omega = data.qvel[3:6].copy()
        gravity = get_gravity_orientation(quat)

        sc = self._obs_scales
        qj_il = np.zeros(na, dtype=np.float32)
        dqj_il = np.zeros(na, dtype=np.float32)
        omega_il = omega * sc.get("base_ang_vel", 0.25)
        for mj_i in range(min(nj, na)):
            il_i = self._mj2il[mj_i]
            qj_il[il_i] = (qj[mj_i] - self._default_angles[mj_i]) * sc.get("dof_pos", 1.0)
            dqj_il[il_i] = dqj[mj_i] * sc.get("dof_vel", 0.05)

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

    def post_step(self, bus):
        """Read action from bus, remap IsaacLab->MuJoCo, return (target, kps, kds)."""
        action_raw = bus.get("action")
        if action_raw is not None:
            action_il = action_raw.flatten()
            self._last_action_il = action_il.copy()

            self._target = self._default_angles.copy()
            for mj_i in range(self._n_actuated):
                il_i = self._mj2il[mj_i]
                self._target[mj_i] = (
                    self._default_angles[mj_i]
                    + action_il[il_i] * self._action_scales[mj_i]
                )

        return (self._target[:self._n_actuated],
                self._kps[:self._n_actuated],
                self._kds[:self._n_actuated])
