"""Utility functions shared across training, evaluation, and UI."""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def binning(value: float, boundaries: Iterable[float]) -> int:
    """Return 0..N bin index where boundaries split [0,1] style ranges."""
    bounds = list(boundaries)
    for idx, b in enumerate(bounds):
        if value < b:
            return idx
    return len(bounds)


def rolling_success_rate(history: list[int], k: int = 100) -> float:
    if not history:
        return 0.0
    window = history[-k:]
    return float(sum(window) / max(1, len(window)))


def epsilon_by_episode(
    episode: int,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
) -> float:
    return max(eps_end, eps_start * (eps_decay**episode))


def save_qtables(walker_q: np.ndarray, mapper_q: np.ndarray, folder: str = "checkpoints") -> None:
    out_dir = ensure_dir(folder)
    np.save(out_dir / "walker_q.npy", walker_q)
    np.save(out_dir / "mapper_q.npy", mapper_q)


def load_qtables(folder: str = "checkpoints") -> tuple[np.ndarray, np.ndarray]:
    in_dir = Path(folder)
    walker_q = np.load(in_dir / "walker_q.npy")
    mapper_q = np.load(in_dir / "mapper_q.npy")
    return walker_q, mapper_q


def write_metrics_csv(rows: list[dict], path: str = "artifacts/metrics.csv") -> None:
    if not rows:
        return
    output = Path(path)
    ensure_dir(output.parent)
    fieldnames_set = set()
    for row in rows:
        fieldnames_set.update(row.keys())
    fieldnames = sorted(fieldnames_set)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass(slots=True)
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else math.nan
