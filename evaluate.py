"""Evaluation, plotting, and ablation utilities for Soulgue-Maze."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

# Keep matplotlib cache in workspace-writable path.
os.environ.setdefault("MPLCONFIGDIR", "artifacts/.mplconfig")
# Force headless-safe backend for script usage (works in Colab/CI/mac terminal).
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import MapperQAgent, WalkerQAgent
from config import DEFAULT_CONFIG, ProjectConfig
from env import SoulgueMazeEnv
from utils import ensure_dir


def _eval_episode(
    env: SoulgueMazeEnv,
    walker: WalkerQAgent,
    mapper: MapperQAgent,
    max_steps: int,
    force_mapper_noop: bool = False,
    use_action_mask: bool = True,
) -> dict:
    obs = env.reset()
    total_w = 0.0
    total_m = 0.0
    success = 0

    for _ in range(max_steps):
        s_w = walker.encode_state(obs.walker_obs)
        s_m = mapper.encode_state(obs.mapper_obs)

        if use_action_mask:
            a_w = walker.select_action(s_w, obs.walker_obs.valid_mask, epsilon=0.0, rng=env.rng)
        else:
            a_w = int(np.argmax(walker.q[s_w]))

        a_m = 0 if force_mapper_noop else mapper.select_action(s_m, epsilon=0.0, rng=env.rng)

        obs, rewards, done, info = env.step(a_w, a_m)
        total_w += rewards.walker_reward
        total_m += rewards.mapper_reward
        if done:
            success = 1
            break

    return {
        "walker_return": total_w,
        "mapper_return": total_m,
        "steps": env.step_count,
        "success": success,
        "accuracy": info["accuracy"],
        "coverage": info["coverage"],
    }


def evaluate_policy(
    cfg: ProjectConfig,
    walker_q: np.ndarray,
    mapper_q: np.ndarray,
    episodes: int = 50,
    seed_offset: int = 5000,
    force_mapper_noop: bool = False,
    use_action_mask: bool = True,
    maze_cfg_override=None,
) -> dict:
    maze_cfg = maze_cfg_override if maze_cfg_override is not None else cfg.maze
    runtime_cfg = replace(cfg.runtime, speed_bonus_horizon=cfg.train.max_steps_per_episode)
    env = SoulgueMazeEnv(maze_cfg=maze_cfg, reward_cfg=cfg.reward, runtime_cfg=runtime_cfg, seed=cfg.train.seed + seed_offset)

    walker = WalkerQAgent(height=maze_cfg.height, width=maze_cfg.width, alpha=cfg.train.alpha, gamma=cfg.train.gamma)
    mapper = MapperQAgent(alpha=cfg.train.alpha, gamma=cfg.train.gamma)
    walker.q[:] = walker_q
    mapper.q[:] = mapper_q

    rows = []
    for ep in range(episodes):
        env.rng = np.random.default_rng(cfg.train.seed + seed_offset + ep)
        row = _eval_episode(
            env=env,
            walker=walker,
            mapper=mapper,
            max_steps=cfg.train.max_steps_per_episode,
            force_mapper_noop=force_mapper_noop,
            use_action_mask=use_action_mask,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    return {
        "episodes": episodes,
        "success_rate": float(df["success"].mean()),
        "avg_steps": float(df["steps"].mean()),
        "avg_accuracy": float(df["accuracy"].mean()),
        "avg_coverage": float(df["coverage"].mean()),
        "walker_return": float(df["walker_return"].mean()),
        "mapper_return": float(df["mapper_return"].mean()),
    }


def run_ablation_suite(
    cfg: ProjectConfig,
    walker_q: np.ndarray,
    mapper_q: np.ndarray,
    episodes: int = 50,
    out_csv: str = "artifacts/ablation_summary.csv",
) -> pd.DataFrame:
    rows = []

    base = evaluate_policy(cfg, walker_q, mapper_q, episodes=episodes)
    rows.append({"setting": "baseline", **base})

    no_bh_maze = replace(cfg.maze, n_blackholes=0)
    no_bh = evaluate_policy(cfg, walker_q, mapper_q, episodes=episodes, maze_cfg_override=no_bh_maze)
    rows.append({"setting": "no_blackhole", **no_bh})

    no_reset = evaluate_policy(cfg, walker_q, mapper_q, episodes=episodes, force_mapper_noop=True)
    rows.append({"setting": "no_reset_action", **no_reset})

    no_mask = evaluate_policy(cfg, walker_q, mapper_q, episodes=episodes, use_action_mask=False)
    rows.append({"setting": "no_action_mask", **no_mask})

    df = pd.DataFrame(rows)
    output = Path(out_csv)
    ensure_dir(output.parent)
    df.to_csv(output, index=False)
    return df


def plot_training_curves(metrics_csv: str = "artifacts/metrics.csv", out_dir: str = "artifacts/plots") -> None:
    df = pd.read_csv(metrics_csv)
    out_path = ensure_dir(out_dir)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(df["episode"], df["walker_return"], alpha=0.65, label="walker")
    plt.plot(df["episode"], df["mapper_return"], alpha=0.65, label="mapper")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episodic Returns")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path / "return_curve.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(df["episode"], df["rolling_success_100"], color="green")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Success@100")
    plt.title("Success Rate")
    plt.tight_layout()
    fig.savefig(out_path / "success_rate.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(df["episode"], df["steps"], color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Avg Steps per Episode")
    plt.tight_layout()
    fig.savefig(out_path / "steps_vs_episode.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(df["episode"], df["coverage"], label="coverage")
    plt.plot(df["episode"], df["accuracy"], label="accuracy")
    plt.xlabel("Episode")
    plt.ylabel("Ratio")
    plt.title("Coverage & Accuracy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path / "coverage_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    from utils import load_qtables

    walker_q, mapper_q = load_qtables("checkpoints")
    summary = evaluate_policy(DEFAULT_CONFIG, walker_q, mapper_q, episodes=50)
    print(summary)
    run_ablation_suite(DEFAULT_CONFIG, walker_q, mapper_q, episodes=50)
    plot_training_curves("artifacts/metrics.csv", "artifacts/plots")
