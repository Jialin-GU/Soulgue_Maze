"""CLI entrypoint for pygame UI demo."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli_config import add_maze_args, cfg_with_maze
from ui import SoulgueUI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Soulgue-Maze pygame UI")
    add_maze_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = cfg_with_maze(args)
    SoulgueUI(cfg).run()
