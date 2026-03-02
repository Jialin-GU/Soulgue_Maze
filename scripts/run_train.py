"""CLI entrypoint for training Soulgue-Maze agents."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG
from cli_config import add_maze_args, maze_from_args
from train import train_full_pipeline, evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Soulgue-Maze tabular Q-learning agents")
    add_maze_args(parser)
    parser.add_argument("--phase-a", type=int, default=DEFAULT_CONFIG.train.phase_a_episodes)
    parser.add_argument("--phase-b", type=int, default=DEFAULT_CONFIG.train.phase_b_episodes)
    parser.add_argument("--phase-c", type=int, default=DEFAULT_CONFIG.train.phase_c_episodes)
    parser.add_argument("--warmup", type=int, default=DEFAULT_CONFIG.train.warmup_6x6_episodes)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.train.seed)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_CONFIG.train.eval_episodes)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        DEFAULT_CONFIG,
        maze=maze_from_args(args),
        train=replace(
            DEFAULT_CONFIG.train,
            seed=args.seed,
            warmup_6x6_episodes=args.warmup,
            phase_a_episodes=args.phase_a,
            phase_b_episodes=args.phase_b,
            phase_c_episodes=args.phase_c,
            eval_episodes=args.eval_episodes,
        ),
    )

    walker, mapper, _rows = train_full_pipeline(cfg)
    summary = evaluate_checkpoint(cfg, walker, mapper)
    print("Final evaluation summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
