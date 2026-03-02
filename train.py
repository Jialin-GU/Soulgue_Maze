"""Training pipeline for dual-agent tabular Q-learning in Soulgue-Maze."""

from __future__ import annotations

from dataclasses import asdict, replace

from config import DEFAULT_CONFIG, MazeConfig, ProjectConfig
from env import SoulgueMazeEnv
from agents import MapperQAgent, WalkerQAgent
from utils import epsilon_by_episode, rolling_success_rate, save_qtables, write_metrics_csv, set_seed


def _build_env(cfg: ProjectConfig, seed: int, maze_cfg: MazeConfig | None = None) -> SoulgueMazeEnv:
    mz = maze_cfg if maze_cfg is not None else cfg.maze
    runtime_cfg = replace(cfg.runtime, speed_bonus_horizon=cfg.train.max_steps_per_episode)
    return SoulgueMazeEnv(
        maze_cfg=mz,
        reward_cfg=cfg.reward,
        runtime_cfg=runtime_cfg,
        seed=seed,
    )


def _run_episode(
    env: SoulgueMazeEnv,
    walker: WalkerQAgent,
    mapper: MapperQAgent,
    epsilon_w: float,
    epsilon_m: float,
    max_steps: int,
    update_walker: bool,
    update_mapper: bool,
    force_mapper_noop: bool,
) -> dict:
    obs = env.reset()
    total_w = 0.0
    total_m = 0.0
    success = 0

    for _ in range(max_steps):
        s_w = walker.encode_state(obs.walker_obs)
        s_m = mapper.encode_state(obs.mapper_obs)

        a_w = walker.select_action(s_w, obs.walker_obs.valid_mask, epsilon_w, env.rng)
        a_m = 0 if force_mapper_noop else mapper.select_action(s_m, epsilon_m, env.rng)

        next_obs, rewards, done, info = env.step(a_w, a_m)

        s_w2 = walker.encode_state(next_obs.walker_obs)
        s_m2 = mapper.encode_state(next_obs.mapper_obs)

        if update_walker:
            walker.update(s_w, a_w, rewards.walker_reward, s_w2, done)
        if update_mapper:
            mapper.update(s_m, a_m, rewards.mapper_reward, s_m2, done)

        total_w += rewards.walker_reward
        total_m += rewards.mapper_reward
        obs = next_obs

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


def train_walker_only(
    cfg: ProjectConfig,
    walker: WalkerQAgent,
    mapper: MapperQAgent,
    episode_offset: int = 0,
) -> tuple[list[dict], int]:
    env = _build_env(cfg, cfg.train.seed + 101)
    rows: list[dict] = []
    global_ep = episode_offset

    success_hist: list[int] = []
    for ep in range(cfg.train.phase_a_episodes):
        eps_w = epsilon_by_episode(ep, cfg.train.eps_start, cfg.train.eps_end, cfg.train.eps_decay)
        metrics = _run_episode(
            env=env,
            walker=walker,
            mapper=mapper,
            epsilon_w=eps_w,
            epsilon_m=0.0,
            max_steps=cfg.train.max_steps_per_episode,
            update_walker=True,
            update_mapper=False,
            force_mapper_noop=True,
        )
        success_hist.append(metrics["success"])

        row = {
            "episode": global_ep,
            "phase": "A_walker",
            "epsilon_w": eps_w,
            "epsilon_m": 0.0,
            "rolling_success_100": rolling_success_rate(success_hist, 100),
            **metrics,
        }
        rows.append(row)
        global_ep += 1

    return rows, global_ep


def warmup_walker_small_maze(
    cfg: ProjectConfig,
    walker: WalkerQAgent,
    mapper: MapperQAgent,
    episode_offset: int,
) -> tuple[list[dict], int]:
    if cfg.train.warmup_6x6_episodes <= 0:
        return [], episode_offset

    warmup_maze = replace(
        cfg.maze,
        height=6,
        width=6,
        n_blackholes=max(1, min(cfg.maze.n_blackholes, 1)),
        maze_pool_size=max(1, min(cfg.maze.maze_pool_size, 3)),
        fixed_layout_seed=cfg.train.seed + 777,
    )
    env = _build_env(cfg, cfg.train.seed + 66, maze_cfg=warmup_maze)

    rows: list[dict] = []
    global_ep = episode_offset
    success_hist: list[int] = []
    max_steps = max(120, int(cfg.train.max_steps_per_episode * 0.75))

    for ep in range(cfg.train.warmup_6x6_episodes):
        eps_w = epsilon_by_episode(ep, cfg.train.eps_start, max(cfg.train.eps_end, 0.08), cfg.train.eps_decay)
        metrics = _run_episode(
            env=env,
            walker=walker,
            mapper=mapper,
            epsilon_w=eps_w,
            epsilon_m=0.0,
            max_steps=max_steps,
            update_walker=True,
            update_mapper=False,
            force_mapper_noop=True,
        )
        success_hist.append(metrics["success"])
        rows.append(
            {
                "episode": global_ep,
                "phase": "W_warmup_6x6",
                "epsilon_w": eps_w,
                "epsilon_m": 0.0,
                "rolling_success_100": rolling_success_rate(success_hist, 100),
                **metrics,
            }
        )
        global_ep += 1
    return rows, global_ep


def train_mapper_with_frozen_walker(
    cfg: ProjectConfig,
    walker: WalkerQAgent,
    mapper: MapperQAgent,
    episode_offset: int,
) -> tuple[list[dict], int]:
    env = _build_env(cfg, cfg.train.seed + 202)
    rows: list[dict] = []
    global_ep = episode_offset

    success_hist: list[int] = []
    frozen_walker_eps = 0.08

    for ep in range(cfg.train.phase_b_episodes):
        eps_m = epsilon_by_episode(ep, cfg.train.eps_start, cfg.train.eps_end, cfg.train.eps_decay)
        metrics = _run_episode(
            env=env,
            walker=walker,
            mapper=mapper,
            epsilon_w=frozen_walker_eps,
            epsilon_m=eps_m,
            max_steps=cfg.train.max_steps_per_episode,
            update_walker=False,
            update_mapper=True,
            force_mapper_noop=False,
        )
        success_hist.append(metrics["success"])

        row = {
            "episode": global_ep,
            "phase": "B_mapper",
            "epsilon_w": frozen_walker_eps,
            "epsilon_m": eps_m,
            "rolling_success_100": rolling_success_rate(success_hist, 100),
            **metrics,
        }
        rows.append(row)
        global_ep += 1

    return rows, global_ep


def alternating_finetune(
    cfg: ProjectConfig,
    walker: WalkerQAgent,
    mapper: MapperQAgent,
    episode_offset: int,
) -> tuple[list[dict], int]:
    env = _build_env(cfg, cfg.train.seed + 303)
    rows: list[dict] = []
    global_ep = episode_offset

    cycle = cfg.train.alt_walker_block + cfg.train.alt_mapper_block
    success_hist: list[int] = []

    for ep in range(cfg.train.phase_c_episodes):
        eps_w = epsilon_by_episode(ep, cfg.train.eps_start * 0.7, cfg.train.eps_end, cfg.train.eps_decay)
        eps_m = epsilon_by_episode(ep, cfg.train.eps_start * 0.7, cfg.train.eps_end, cfg.train.eps_decay)

        slot = ep % max(1, cycle)
        update_walker = slot < cfg.train.alt_walker_block
        update_mapper = not update_walker

        metrics = _run_episode(
            env=env,
            walker=walker,
            mapper=mapper,
            epsilon_w=eps_w,
            epsilon_m=eps_m,
            max_steps=cfg.train.max_steps_per_episode,
            update_walker=update_walker,
            update_mapper=update_mapper,
            force_mapper_noop=False,
        )
        success_hist.append(metrics["success"])

        row = {
            "episode": global_ep,
            "phase": "C_alt",
            "epsilon_w": eps_w,
            "epsilon_m": eps_m,
            "update_walker": int(update_walker),
            "update_mapper": int(update_mapper),
            "rolling_success_100": rolling_success_rate(success_hist, 100),
            **metrics,
        }
        rows.append(row)
        global_ep += 1

    return rows, global_ep


def evaluate_checkpoint(
    cfg: ProjectConfig,
    walker: WalkerQAgent,
    mapper: MapperQAgent,
    episodes: int | None = None,
) -> dict:
    from evaluate import evaluate_policy

    result = evaluate_policy(
        cfg=cfg,
        walker_q=walker.q,
        mapper_q=mapper.q,
        episodes=episodes or cfg.train.eval_episodes,
        seed_offset=5000,
    )
    return result


def train_full_pipeline(cfg: ProjectConfig = DEFAULT_CONFIG) -> tuple[WalkerQAgent, MapperQAgent, list[dict]]:
    set_seed(cfg.train.seed)

    walker = WalkerQAgent(
        height=cfg.maze.height,
        width=cfg.maze.width,
        alpha=cfg.train.alpha,
        gamma=cfg.train.gamma,
    )
    mapper = MapperQAgent(alpha=cfg.train.alpha, gamma=cfg.train.gamma)

    all_rows: list[dict] = []
    ep_cursor = 0

    rows_w, ep_cursor = warmup_walker_small_maze(cfg, walker, mapper, ep_cursor)
    all_rows.extend(rows_w)

    rows_a, ep_cursor = train_walker_only(cfg, walker, mapper, ep_cursor)
    all_rows.extend(rows_a)

    rows_b, ep_cursor = train_mapper_with_frozen_walker(cfg, walker, mapper, ep_cursor)
    all_rows.extend(rows_b)

    rows_c, ep_cursor = alternating_finetune(cfg, walker, mapper, ep_cursor)
    all_rows.extend(rows_c)

    write_metrics_csv(all_rows, path="artifacts/metrics.csv")
    save_qtables(walker.q, mapper.q, folder="checkpoints")

    return walker, mapper, all_rows


def config_to_dict(cfg: ProjectConfig = DEFAULT_CONFIG) -> dict:
    return {
        "maze": asdict(cfg.maze),
        "reward": asdict(cfg.reward),
        "train": asdict(cfg.train),
        "runtime": asdict(cfg.runtime),
    }


if __name__ == "__main__":
    train_full_pipeline(DEFAULT_CONFIG)
