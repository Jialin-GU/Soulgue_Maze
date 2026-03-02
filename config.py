"""Centralized configuration for Soulgue-Maze."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MazeConfig:
    height: int = 8
    width: int = 8
    n_entries: int = 2
    n_blackholes: int = 1
    fixed_layout_seed: int | None = None
    maze_pool_size: int = 3
    maze_pool_seed: int = 2026
    connectivity_carve_extra_prob: float = 0.08
    blackhole_min_entry_distance: int = 2
    blackhole_min_reachable_ratio: float = 0.85
    blackhole_relaxed_reachable_ratio: float = 0.70
    accuracy_target: float = 1.0


@dataclass(slots=True)
class RewardConfig:
    walker_new_cell: float = 1.6
    walker_new_edge: float = 0.7
    walker_wall_hit: float = -0.3
    walker_step_cost: float = -0.03
    walker_backtrack_penalty: float = -1.0
    walker_loop_penalty: float = -0.7
    walker_blackhole_penalty: float = -1.0
    walker_done_bonus: float = 5.0

    mapper_step_cost: float = -0.01
    mapper_accuracy_gain_scale: float = 24.0
    mapper_new_edge: float = 0.5
    mapper_reset_cost: float = -0.5
    mapper_bad_reset_penalty: float = -1.5
    mapper_noop_stuck_penalty: float = -0.2
    mapper_reset_recovery_bonus: float = 1.5
    mapper_done_bonus: float = 8.0
    team_fast_finish_bonus_scale: float = 10.0


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    alpha: float = 0.15
    gamma: float = 0.97

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.9985

    max_steps_per_episode: int = 420
    eval_interval: int = 100
    eval_episodes: int = 50

    warmup_6x6_episodes: int = 800
    phase_a_episodes: int = 2000
    phase_b_episodes: int = 1600
    phase_c_episodes: int = 2400

    alt_walker_block: int = 200
    alt_mapper_block: int = 100


@dataclass(slots=True)
class EnvRuntimeConfig:
    max_resets_per_episode: int = 6
    stuck_window: int = 14
    severe_stuck_unique_threshold: int = 3
    mild_stuck_unique_threshold: int = 5
    reset_recovery_window: int = 10
    speed_bonus_horizon: int = 420


@dataclass(slots=True)
class UIConfig:
    cell_px: int = 52
    margin_px: int = 22
    view_gap_px: int = 26
    panel_width_px: int = 340
    fps: int = 10
    fog_alpha: int = 145
    ai_walker_epsilon: float = 0.14
    ai_mapper_epsilon: float = 0.05
    stuck_reset_guard_steps: int = 8
    oscillation_window: int = 8
    show_truth_walls_default: bool = False
    show_truth_blackholes_default: bool = True
    mapper_show_true_blackholes: bool = False


@dataclass(slots=True)
class ProjectConfig:
    maze: MazeConfig = field(default_factory=MazeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: EnvRuntimeConfig = field(default_factory=EnvRuntimeConfig)
    ui: UIConfig = field(default_factory=UIConfig)


DEFAULT_CONFIG = ProjectConfig()
