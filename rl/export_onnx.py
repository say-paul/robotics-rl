"""Export a trained SB3 PPO policy to ONNX with baked-in observation normalisation.

The resulting ``.onnx`` file is directly loadable by the existing
``OnnxPolicyAction`` in ``actions/policy.py`` with no code changes.

Usage
-----
    python -m rl.export_onnx \\
        --model rl/checkpoints/best_model.zip \\
        --output policies/g1_walk_ppo.onnx

    python -m rl.export_onnx \\
        --model rl/checkpoints/best_model.zip \\
        --stage stand --output policies/g1_stand.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.configs.training_config import TrainingConfig
from rl.envs.g1_walk_env import G1WalkEnv, OBS_DIM

log = logging.getLogger(__name__)


class NormalizedPolicy(nn.Module):
    """Wraps the SB3 actor with observation normalisation so the ONNX graph
    includes the running-mean / running-var statistics and outputs
    deterministic (mean) actions clipped to [-1, 1] (matching SB3 predict)."""

    def __init__(
        self,
        actor: nn.Module,
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
        clip_obs: float = 10.0,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_var", obs_var)
        self.clip_obs = clip_obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normed = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        normed = torch.clamp(normed, -self.clip_obs, self.clip_obs)
        action = self.actor(normed)
        return torch.clamp(action, -1.0, 1.0)


def export(
    model_path: str,
    output_path: str,
    cfg: TrainingConfig | None = None,
    clip_obs: float = 10.0,
) -> None:
    """Load a trained model and export to ONNX."""
    cfg = cfg or TrainingConfig()
    raw_env = DummyVecEnv([lambda: G1WalkEnv(env_cfg=cfg.env, reward_cfg=cfg.reward)])

    vecnorm_path = Path(model_path).with_suffix(".vecnorm.pkl")
    if not vecnorm_path.exists():
        candidates = list(Path(model_path).parent.glob("*.vecnorm.pkl"))
        vecnorm_path = candidates[0] if candidates else vecnorm_path

    model = PPO.load(model_path)

    # Extract normalisation stats
    has_vecnorm = vecnorm_path.exists()
    if has_vecnorm:
        log.info("Loading VecNormalize from %s", vecnorm_path)
        vec_env = VecNormalize.load(str(vecnorm_path), raw_env)
        obs_mean = torch.tensor(vec_env.obs_rms.mean, dtype=torch.float32)
        obs_var = torch.tensor(vec_env.obs_rms.var, dtype=torch.float32)
        vec_env.close()
    else:
        log.warning("No VecNormalize stats found; exporting without normalisation")
        obs_mean = torch.zeros(OBS_DIM, dtype=torch.float32)
        obs_var = torch.ones(OBS_DIM, dtype=torch.float32)

    raw_env.close()

    wrapped = _build_actor_forward(model, obs_mean, obs_var, clip_obs)
    wrapped.eval()

    dummy = torch.randn(1, OBS_DIM)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy,
        output_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Numerical sanity check
    _validate(model_path, vecnorm_path if has_vecnorm else None, output_path, cfg=cfg)

    log.info("Exported ONNX model to %s", output_path)


class _SB3ActorWrapper(nn.Module):
    """Extracts the deterministic action from an SB3 MlpPolicy."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(features)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)


def _build_actor_forward(
    model: PPO,
    obs_mean: torch.Tensor,
    obs_var: torch.Tensor,
    clip_obs: float,
) -> NormalizedPolicy:
    actor = _SB3ActorWrapper(model.policy)
    return NormalizedPolicy(actor, obs_mean, obs_var, clip_obs)


def _validate(
    model_path: str,
    vecnorm_path: Path | None,
    onnx_path: str,
    cfg: TrainingConfig | None = None,
    n_samples: int = 5,
) -> None:
    """Compare ONNX outputs against SB3 inference.

    The ONNX model has normalisation baked in and expects *raw* obs.
    SB3's model.predict receives already-normalised obs from VecNormalize.
    We feed each pathway accordingly and compare the resulting actions.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cfg = cfg or TrainingConfig()
    raw_env = DummyVecEnv([lambda: G1WalkEnv(env_cfg=cfg.env, reward_cfg=cfg.reward)])
    if vecnorm_path and vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=False)

    model = PPO.load(model_path, env=env)

    norm_obs = env.reset()  # normalised
    max_err = 0.0
    for _ in range(n_samples):
        raw_obs = env.get_original_obs()  # un-normalised
        sb3_action, _ = model.predict(norm_obs, deterministic=True)
        onnx_out = session.run(None, {input_name: raw_obs.astype(np.float32)})[0]
        err = float(np.max(np.abs(sb3_action - onnx_out)))
        max_err = max(max_err, err)
        norm_obs, _, _, _ = env.step(sb3_action)

    env.close()
    log.info("ONNX validation max abs error: %.6e", max_err)
    if max_err > 1e-4:
        log.warning("Large discrepancy between SB3 and ONNX outputs!")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export PPO policy to ONNX")
    parser.add_argument("--model", required=True, help="Path to .zip model")
    parser.add_argument(
        "--stage",
        choices=list(TrainingConfig.STAGES),
        default=None,
        help="Curriculum stage config (stand, slow_walk, full_walk)",
    )
    parser.add_argument("--output", default="policies/g1_walk_ppo.onnx")
    args = parser.parse_args()

    cfg = TrainingConfig.from_stage(args.stage)
    export(args.model, args.output, cfg=cfg)


if __name__ == "__main__":
    main()
