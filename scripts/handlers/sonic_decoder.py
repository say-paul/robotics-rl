"""Handler for the SONIC policy decoder.

Builds the 994-dim policy observation from bus signals:
    token_state(64) + ang_vel_hist(30) + jpos_hist(290) +
    jvel_hist(290) + action_hist(290) + gravity_hist(30) = 994

GraphPolicyRunner.step() is responsible for writing the history arrays
to the bus each tick; this handler just reads and concatenates them.

Usage in YAML:
    nodes:
      - name: "policy_inference"
        model: "sonic_policy_decoder"
        handler: "handlers.sonic_decoder.build_policy_obs"
"""

import numpy as np

POLICY_OBS_DIM = 994


def build_policy_obs(bus, model_cfg, node_cfg):
    """Build the 994-dim decoder input from bus signals.

    Expected bus signals:
        - "encoded_tokens": shape [64], output from encoder
        - "ang_vel_hist":   shape [30], flattened (10 frames × 3)
        - "jpos_hist":      shape [290], flattened (10 frames × 29)
        - "jvel_hist":      shape [290], flattened (10 frames × 29)
        - "action_hist":    shape [290], flattened (10 frames × 29)
        - "gravity_hist":   shape [30], flattened (10 frames × 3)

    Returns:
        dict mapping the ONNX input name to a [1, 994] float32 array.
        Uses zero tokens when encoded_tokens is missing (idle/no reference),
        matching the C++ reference which keeps token_state at zeros.
    """
    token = bus.get("encoded_tokens")
    if token is None:
        token = np.zeros(64, dtype=np.float32)
    else:
        token = token.flatten()

    ang_vel = bus.get("ang_vel_hist")
    jpos = bus.get("jpos_hist")
    jvel = bus.get("jvel_hist")
    actions = bus.get("action_hist")
    gravity = bus.get("gravity_hist")

    parts = [token]
    for arr, expected_size in [
        (ang_vel, 30), (jpos, 290), (jvel, 290),
        (actions, 290), (gravity, 30),
    ]:
        if arr is not None:
            parts.append(arr.flatten())
        else:
            parts.append(np.zeros(expected_size, dtype=np.float32))

    obs = np.concatenate(parts).astype(np.float32)

    if obs.shape[0] != POLICY_OBS_DIM:
        padded = np.zeros(POLICY_OBS_DIM, dtype=np.float32)
        n = min(obs.shape[0], POLICY_OBS_DIM)
        padded[:n] = obs[:n]
        obs = padded

    return {"obs_dict": obs.reshape(1, -1)}
