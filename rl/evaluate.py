"""Evaluate a trained PPO policy and optionally record video.

Usage
-----
    python -m rl.evaluate --model rl/checkpoints/best_model.zip --episodes 20
    python -m rl.evaluate --model rl/checkpoints/best_model.zip --stage stand --episodes 10
    python -m rl.evaluate --model rl/checkpoints/best_model.zip --render --video out.mp4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.configs.training_config import TrainingConfig
from rl.envs.g1_walk_env import G1WalkEnv

log = logging.getLogger(__name__)


def _load_model_and_env(
    model_path: str, cfg: TrainingConfig
) -> tuple[PPO, VecNormalize]:
    """Load model + VecNormalize stats (if present)."""
    raw_env = DummyVecEnv([lambda: G1WalkEnv(env_cfg=cfg.env, reward_cfg=cfg.reward)])

    vecnorm_path = Path(model_path).with_suffix(".vecnorm.pkl")
    if not vecnorm_path.exists():
        parent = Path(model_path).parent
        vecnorm_candidates = sorted(
            list(parent.glob("*.vecnorm.pkl"))
            + list(parent.glob("*vecnormalize*.pkl")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if vecnorm_candidates:
            vecnorm_path = vecnorm_candidates[0]

    if vecnorm_path.exists():
        log.info("Loading VecNormalize from %s", vecnorm_path)
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training = False
        env.norm_reward = False
    else:
        log.warning("No VecNormalize found; using raw observations")
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=False)

    model = PPO.load(model_path, env=env)
    return model, env


def evaluate(
    model: PPO,
    env: VecNormalize,
    n_episodes: int = 20,
    deterministic: bool = True,
    render: bool = False,
    video_path: Optional[str] = None,
) -> dict:
    """Run *n_episodes* and collect statistics."""
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_fwd_vels: List[float] = []
    falls = 0

    frames: List[np.ndarray] = []
    record = render and video_path is not None

    obs = env.reset()
    ep_reward = 0.0
    ep_len = 0
    fwd_vels: List[float] = []

    while len(episode_rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = env.step(action)

        ep_reward += float(reward[0])
        ep_len += 1

        info = infos[0]
        fwd_vels.append(info.get("forward_vel", 0.0))

        if record and len(episode_rewards) == 0:
            inner_env: G1WalkEnv = env.venv.envs[0]  # type: ignore[attr-defined]
            frame = inner_env.render()
            if frame is not None:
                frames.append(frame)

        if done[0]:
            terminal_info = info.get("terminal_observation") is not None
            if info.get("TimeLimit.truncated", False) is False and terminal_info:
                falls += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            episode_fwd_vels.append(float(np.mean(fwd_vels)) if fwd_vels else 0.0)

            ep_reward = 0.0
            ep_len = 0
            fwd_vels = []

    stats = {
        "episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_fwd_vel": float(np.mean(episode_fwd_vels)),
        "fall_rate": falls / n_episodes,
    }

    if record and frames:
        _save_video(frames, video_path)  # type: ignore[arg-type]

    return stats


def _save_video(frames: List[np.ndarray], path: str, fps: int = 50) -> None:
    """Write frames to an MP4 file using imageio."""
    try:
        import imageio.v3 as iio
    except ImportError:
        log.warning("imageio not installed; skipping video save")
        return
    log.info("Writing %d frames to %s", len(frames), path)
    iio.imwrite(path, np.stack(frames), fps=fps, codec="h264")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate a trained G1 walking policy")
    parser.add_argument("--model", required=True, help="Path to .zip model")
    parser.add_argument(
        "--stage",
        choices=list(TrainingConfig.STAGES),
        default=None,
        help="Curriculum stage config (stand, slow_walk, full_walk)",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true", help="Render first episode")
    parser.add_argument("--video", type=str, default=None, help="Save video to path")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    cfg = TrainingConfig.from_stage(args.stage)
    model, env = _load_model_and_env(args.model, cfg)

    stats = evaluate(
        model,
        env,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        video_path=args.video,
    )

    print("\n=== Evaluation Results ===")
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")

    env.close()


if __name__ == "__main__":
    main()
