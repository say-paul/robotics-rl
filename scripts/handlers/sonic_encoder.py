"""Reference handler for the SONIC observation encoder.

This handler builds the 1762-dim encoder observation vector from bus
signals, using standing-reference fields (zeros for joint positions,
standing height for root z, 6D rotation for anchor orientation).

Extracted from policy_runners.SonicPolicyRunner._build_encoder_obs so
that the generic model_runner can drive the SONIC encoder without
hard-coded runner logic.

Usage in YAML:
    nodes:
      - name: "encoder_inference"
        model: "sonic_encoder"
        handler: "handlers.sonic_encoder.build_obs"
"""

import numpy as np

# Encoder observation layout — field name and dimensionality.
# Total: 1762 dims.
ENCODER_OBS_LAYOUT = [
    ("encoder_mode_4",                              4),
    ("motion_joint_positions_10frame_step5",       290),
    ("motion_joint_velocities_10frame_step5",      290),
    ("motion_root_z_position_10frame_step5",        10),
    ("motion_root_z_position",                       1),
    ("motion_anchor_orientation",                    6),
    ("motion_anchor_orientation_10frame_step5",     60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target",                       9),
    ("vr_3point_local_orn_target",                  12),
    ("smpl_joints_10frame_step1",                  720),
    ("smpl_anchor_orientation_10frame_step1",       60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
]
ENCODER_TOTAL_DIM = sum(d for _, d in ENCODER_OBS_LAYOUT)  # 1762


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _quat_to_rot_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _anchor_orientation_6d(base_quat, ref_quat=None):
    """6D rotation representation: first 2 columns of rotation matrix."""
    if ref_quat is None:
        ref_quat = np.array([1.0, 0.0, 0.0, 0.0])
    rel = _quat_multiply(_quat_conjugate(base_quat), ref_quat)
    R = _quat_to_rot_matrix(rel)
    return np.array([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]],
                    dtype=np.float32)


def build_obs(bus, model_cfg, node_cfg):
    """Build the 1762-dim encoder observation from bus signals.

    Expected bus signals:
        - "base_quaternion": shape [4], robot torso orientation (w, x, y, z)
        - "standing_height": shape [1], robot standing z-position
        - "encoder_mode": shape [1], int index into the 4-dim one-hot

    Returns:
        dict mapping the ONNX input name to a [1, 1762] float32 array,
        or None if required signals are missing.
    """
    quat = bus.get("base_quaternion")
    standing_h = bus.get("standing_height")
    enc_mode_arr = bus.get("encoder_mode")

    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if standing_h is None:
        standing_h = np.array([0.75], dtype=np.float32)
    encoder_mode = int(enc_mode_arr[0]) if enc_mode_arr is not None else 0

    standing_height = float(standing_h.flat[0])
    orn_6 = _anchor_orientation_6d(quat)

    obs = np.zeros(ENCODER_TOTAL_DIM, dtype=np.float32)
    offset = 0
    for name, dim in ENCODER_OBS_LAYOUT:
        if name == "encoder_mode_4":
            obs[offset + encoder_mode] = 1.0
        elif name == "motion_root_z_position_10frame_step5":
            obs[offset:offset + dim] = standing_height
        elif name == "motion_root_z_position":
            obs[offset] = standing_height
        elif name == "motion_anchor_orientation":
            obs[offset:offset + dim] = orn_6
        elif name == "motion_anchor_orientation_10frame_step5":
            obs[offset:offset + dim] = np.tile(orn_6, 10)
        offset += dim

    return {"obs_dict": obs.reshape(1, -1)}
