"""Shared CLI config helpers for train/eval/ui scripts."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import replace

from config import DEFAULT_CONFIG, MazeConfig, ProjectConfig


def add_maze_args(parser: ArgumentParser) -> None:
    parser.add_argument("--height", type=int, default=DEFAULT_CONFIG.maze.height)
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG.maze.width)
    parser.add_argument("--maze-pool-size", type=int, default=DEFAULT_CONFIG.maze.maze_pool_size)
    parser.add_argument("--maze-pool-seed", type=int, default=DEFAULT_CONFIG.maze.maze_pool_seed)
    parser.add_argument("--fixed-layout-seed", type=int, default=-1, help="-1 means disabled")
    parser.add_argument("--accuracy-target", type=float, default=DEFAULT_CONFIG.maze.accuracy_target)


def maze_from_args(args: Namespace) -> MazeConfig:
    fixed_seed = None if int(args.fixed_layout_seed) < 0 else int(args.fixed_layout_seed)
    return replace(
        DEFAULT_CONFIG.maze,
        height=int(args.height),
        width=int(args.width),
        maze_pool_size=int(args.maze_pool_size),
        maze_pool_seed=int(args.maze_pool_seed),
        fixed_layout_seed=fixed_seed,
        accuracy_target=float(args.accuracy_target),
    )


def cfg_with_maze(args: Namespace) -> ProjectConfig:
    return replace(DEFAULT_CONFIG, maze=maze_from_args(args))
