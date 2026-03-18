"""Import a pretrained ONNX locomotion policy into an SB3 PPO model for fine-tuning.

Supports policies exported from mujoco_playground (e.g. ``g1_joystick.onnx``)
that follow the 103-dim obs / 29-dim action convention.

The ONNX graph typically contains:
  1. Observation normalisation (Sub + Mul) with baked-in mean / reciprocal-std
  2. An MLP with SiLU (Swish) activations
  3. A final linear layer whose output is split into [action_mean, action_log_std]

This module extracts the normalisation stats and MLP weights, builds a matching
SB3 ``PPO`` model, and transplants the weights so that the resulting model
produces identical outputs to the original ONNX file.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import onnx
import onnx.numpy_helper
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

log = logging.getLogger(__name__)


def _extract_onnx_weights(onnx_path: str) -> dict[str, np.ndarray]:
    """Load all named initializers from an ONNX model."""
    model = onnx.load(onnx_path)
    return {
        init.name: onnx.numpy_helper.to_array(init)
        for init in model.graph.initializer
    }


def _parse_sb3_export_weights(
    weights: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], int, float | None]:
    """Parse ONNX exported by rl.export_onnx (actor.mlp_extractor.policy_net.*, action_net).

    Returns (obs_mean, obs_reciprocal_std, layer_list, action_dim, log_std_init).
    log_std_init is None if the ONNX has mean+log_std; otherwise a float for action head only.
    """
    if "actor.mlp_extractor.policy_net.0.weight" not in weights:
        return None
    obs_mean = weights.get("obs_mean")
    sqrt_var = weights.get("sqrt")
    if obs_mean is None:
        obs_mean = np.zeros(103, dtype=np.float32)
    if sqrt_var is not None and sqrt_var.size == 103:
        obs_recip_std = 1.0 / (np.asarray(sqrt_var, dtype=np.float64) + 1e-8)
    else:
        obs_recip_std = np.ones(103, dtype=np.float32)

    layers = []
    for i in [0, 2, 4]:
        w = weights.get(f"actor.mlp_extractor.policy_net.{i}.weight")
        b = weights.get(f"actor.mlp_extractor.policy_net.{i}.bias")
        if w is None or b is None:
            break
        layers.append((w, b))
    aw = weights.get("actor.action_net.weight")
    ab = weights.get("actor.action_net.bias")
    if aw is not None and ab is not None:
        layers.append((aw, ab))
    if not layers:
        return None
    action_dim = int(layers[-1][1].shape[0])
    return obs_mean, obs_recip_std, layers, action_dim, -2.0


def _parse_joystick_weights(
    weights: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], int] | None:
    """Parse the g1_joystick ONNX weight layout.

    Returns (obs_mean, obs_reciprocal_std, layer_list, action_dim) or None if not matched.
    """
    keys = sorted(weights.keys())
    obs_mean = None
    obs_recip_std = None
    layers: list[tuple[np.ndarray, np.ndarray]] = []

    for key in keys:
        if "sub/ReadVariableOp" in key:
            obs_mean = weights[key]
        elif "truediv_recip" in key or "truediv" in key:
            obs_recip_std = weights[key]

    hidden_w_keys = sorted(
        [k for k in keys if "/Cast/ReadVariableOp" in k or ("/MatMul" in k and "hidden" in k)],
    )
    hidden_b_keys = sorted(
        [k for k in keys if "/BiasAdd/ReadVariableOp" in k],
    )
    if not hidden_w_keys or not hidden_b_keys:
        return None

    for wk, bk in zip(hidden_w_keys, hidden_b_keys):
        w = weights[wk]
        b = weights[bk]
        layers.append((w, b))
    if not layers:
        return None
    last_out = layers[-1][1].shape[0]
    action_dim = last_out // 2
    if obs_mean is None:
        obs_mean = np.zeros(103, dtype=np.float32)
    if obs_recip_std is None:
        obs_recip_std = np.ones(103, dtype=np.float32)
    return obs_mean, obs_recip_std, layers, action_dim


def import_onnx_into_ppo(
    onnx_path: str,
    env: VecEnv | VecNormalize,
    device: torch.device | str = "cpu",
    **ppo_kwargs,
) -> PPO:
    """Create an SB3 PPO model pre-initialised from an ONNX locomotion policy.

    The architecture is matched to the ONNX graph (SiLU activations, same
    layer widths). The observation normalisation from the ONNX graph is loaded
    into the ``VecNormalize`` wrapper so SB3's pipeline handles it natively.
    """
    weights = _extract_onnx_weights(onnx_path)
    log_std_init = None
    parsed = _parse_sb3_export_weights(weights)
    if parsed is not None:
        obs_mean, obs_recip_std, layers, action_dim, log_std_init = parsed
        log.info("Detected SB3-exported ONNX format (actor.policy_net + action_net)")
    else:
        joystick = _parse_joystick_weights(weights)
        if joystick is None:
            raise ValueError(
                "ONNX format not recognized. Expected either SB3 export "
                "(actor.mlp_extractor.policy_net.*, action_net) or g1_joystick layout."
            )
        obs_mean, obs_recip_std, layers, action_dim = joystick
    obs_var = (1.0 / (np.asarray(obs_recip_std, dtype=np.float64) + 1e-8)) ** 2

    # Determine hidden layer sizes (excluding the final output layer)
    hidden_sizes = [layer[1].shape[0] for layer in layers[:-1]]
    log.info(
        "Pretrained architecture: %s → %s → %d (SiLU activations)",
        layers[0][0].shape[0],
        " → ".join(str(s) for s in hidden_sizes),
        action_dim,
    )

    # Inject normalisation stats into VecNormalize if present
    if isinstance(env, VecNormalize):
        env.obs_rms.mean[:] = obs_mean
        env.obs_rms.var[:] = obs_var
        env.obs_rms.count = 1_000_000  # high count so stats don't shift quickly
        log.info("Loaded observation normalisation into VecNormalize")

    # Use log_std_init from config if provided (same as scratch/resume path)
    ppo_kwargs = dict(ppo_kwargs)
    policy_kwargs = dict(
        net_arch=dict(pi=hidden_sizes, vf=hidden_sizes),
        activation_fn=nn.SiLU,
        log_std_init=ppo_kwargs.pop("log_std_init", log_std_init if log_std_init is not None else -1.0),
    )
    # Build PPO with matching architecture (same policy_kwargs shape as scratch path)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device=device,
        **ppo_kwargs,
    )

    # Transplant weights into the SB3 policy (log_std_init used when ONNX has action mean only)
    _load_actor_weights(model, layers, action_dim, log_std_init=log_std_init)

    log.info("Imported pretrained weights from %s", onnx_path)
    return model


def _load_actor_weights(
    model: PPO,
    layers: list[tuple[np.ndarray, np.ndarray]],
    action_dim: int,
    log_std_init: float | None = None,
) -> None:
    """Copy ONNX layer weights into the SB3 policy's actor MLP.

    If log_std_init is set (e.g. from SB3 export with action mean only), policy
    log_std is set to that scalar; otherwise the last layer bias is split into
    mean and log_std (joystick format).
    """
    policy = model.policy
    dev = next(policy.parameters()).device
    mlp = policy.mlp_extractor.policy_net

    # Hidden layers: ONNX often has (out, in), PyTorch Linear is (out, in); SB3 export is (out, in)
    linear_idx = 0
    for module in mlp.modules():
        if not isinstance(module, nn.Linear):
            continue
        if linear_idx >= len(layers) - 1:
            break
        w, b = layers[linear_idx]
        # SB3 export: weight shape (out, in); transpose to match if needed
        if w.shape[0] == b.shape[0]:
            module.weight.data = torch.tensor(w, dtype=torch.float32, device=dev)
        else:
            module.weight.data = torch.tensor(w.T, dtype=torch.float32, device=dev)
        module.bias.data = torch.tensor(b, dtype=torch.float32, device=dev)
        linear_idx += 1

    final_w, final_b = layers[-1]
    if log_std_init is not None:
        mean_w = final_w
        mean_b = final_b
        if mean_w.shape[0] != action_dim:
            mean_w = mean_w.T
        policy.action_net.weight.data = torch.tensor(mean_w, dtype=torch.float32, device=dev)
        policy.action_net.bias.data = torch.tensor(mean_b, dtype=torch.float32, device=dev)
        if hasattr(policy, "log_std"):
            policy.log_std.data.fill_(log_std_init)
    else:
        mean_w = final_w[:, :action_dim]
        mean_b = final_b[:action_dim]
        log_std_b = final_b[action_dim:]
        policy.action_net.weight.data = torch.tensor(mean_w.T, dtype=torch.float32, device=dev)
        policy.action_net.bias.data = torch.tensor(mean_b, dtype=torch.float32, device=dev)
        if hasattr(policy, "log_std"):
            policy.log_std.data = torch.tensor(log_std_b, dtype=torch.float32, device=dev)

    log.info(
        "Transplanted %d layers + action head into SB3 policy",
        linear_idx + 1,
    )
