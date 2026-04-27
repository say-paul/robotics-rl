"""Reference handler for the SONIC observation encoder.

Builds the 1762-dim encoder observation vector from bus signals.

When the planner is active and reference trajectory data is on the bus
(written by SonicWBCHandler._extract_reference_trajectory), the handler
uses the planner's reference motion for joint positions, velocities,
root-z, and anchor orientation.  Otherwise falls back to standing-
reference defaults.

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

N_LOWER_BODY = 12


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

    Expected bus signals (always present):
        - "base_quaternion": shape [4], robot torso orientation (w, x, y, z)
        - "standing_height": shape [1], robot standing z-position
        - "encoder_mode": shape [1], int index into the 4-dim one-hot

    Optional bus signals (written by SonicWBCHandler when planner active):
        - "ref_joint_positions_10f":  [10, 29] IsaacLab order
        - "ref_joint_velocities_10f": [10, 29]
        - "ref_root_z_10f":           [10]
        - "ref_orientation_10f":      [10, 4]  quaternion (w,x,y,z)
        - "ref_root_z":               [1]
        - "ref_orientation":          [4]

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

    mode_arr = bus.get("mode")
    current_mode = int(mode_arr.flat[0]) if mode_arr is not None else 0

    ref_jpos = bus.get("ref_joint_positions_10f")
    ref_jvel = bus.get("ref_joint_velocities_10f")
    ref_rz10 = bus.get("ref_root_z_10f")
    ref_rz = bus.get("ref_root_z")
    ref_orn = bus.get("ref_orientation")
    ref_orn10 = bus.get("ref_orientation_10f")

    mv_dir = bus.get("movement_direction")
    has_movement = mv_dir is not None and np.any(np.abs(mv_dir) > 1e-6)
    has_ref = ref_jpos is not None and current_mode != 0 and has_movement

    if has_ref and ref_orn is not None:
        orn_6 = _anchor_orientation_6d(quat, ref_orn)
    else:
        orn_6 = _anchor_orientation_6d(quat)

    obs = np.zeros(ENCODER_TOTAL_DIM, dtype=np.float32)
    offset = 0
    for name, dim in ENCODER_OBS_LAYOUT:
        if name == "encoder_mode_4":
            obs[offset + encoder_mode] = 1.0

        elif name == "motion_joint_positions_10frame_step5":
            if has_ref:
                obs[offset:offset + dim] = ref_jpos.flatten()[:dim]
            # else: zeros (standing reference)

        elif name == "motion_joint_velocities_10frame_step5":
            if has_ref and ref_jvel is not None:
                obs[offset:offset + dim] = ref_jvel.flatten()[:dim]

        elif name == "motion_root_z_position_10frame_step5":
            if has_ref and ref_rz10 is not None:
                obs[offset:offset + dim] = ref_rz10.flatten()[:dim]
            else:
                obs[offset:offset + dim] = standing_height

        elif name == "motion_root_z_position":
            if has_ref and ref_rz is not None:
                obs[offset] = float(ref_rz.flat[0])
            else:
                obs[offset] = standing_height

        elif name == "motion_anchor_orientation":
            obs[offset:offset + dim] = orn_6

        elif name == "motion_anchor_orientation_10frame_step5":
            if has_ref and ref_orn10 is not None:
                for fi in range(10):
                    rq = ref_orn10[fi] if fi < len(ref_orn10) else ref_orn10[-1]
                    obs[offset + fi * 6:offset + fi * 6 + 6] = \
                        _anchor_orientation_6d(quat, rq)
            else:
                obs[offset:offset + dim] = np.tile(orn_6, 10)

        elif name == "motion_joint_positions_lowerbody_10frame_step5":
            if has_ref:
                lb = ref_jpos[:, :N_LOWER_BODY]
                obs[offset:offset + dim] = lb.flatten()[:dim]

        elif name == "motion_joint_velocities_lowerbody_10frame_step5":
            if has_ref and ref_jvel is not None:
                lb = ref_jvel[:, :N_LOWER_BODY]
                obs[offset:offset + dim] = lb.flatten()[:dim]

        offset += dim

    return {"obs_dict": obs.reshape(1, -1)}
