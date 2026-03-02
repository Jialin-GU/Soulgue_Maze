"""CLI entrypoint for evaluation and plotting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli_config import add_maze_args, cfg_with_maze
from evaluate import evaluate_policy, plot_training_curves, run_ablation_suite
from utils import load_qtables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Soulgue-Maze checkpoints")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--metrics-csv", type=str, default="artifacts/metrics.csv")
    parser.add_argument("--plots-dir", type=str, default="artifacts/plots")
    add_maze_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = cfg_with_maze(args)
    walker_q, mapper_q = load_qtables(args.checkpoints)

    summary = evaluate_policy(cfg, walker_q, mapper_q, episodes=args.episodes)
    print("Policy summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    ablation = run_ablation_suite(cfg, walker_q, mapper_q, episodes=args.episodes)
    print("\nAblation summary:")
    print(ablation)

    plot_training_curves(args.metrics_csv, args.plots_dir)
    print(f"\nPlots generated under {args.plots_dir}")


if __name__ == "__main__":
    main()
