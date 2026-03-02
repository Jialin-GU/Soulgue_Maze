"""Environment and data models for Soulgue-Maze."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from config import MazeConfig, RewardConfig, EnvRuntimeConfig

# Action mapping used across project.
# 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
ACTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
N_WALKER_ACTIONS = 4
N_MAPPER_ACTIONS = 3  # NOOP / RESET_ENTRY_0 / RESET_ENTRY_1

UNKNOWN = -1
FREE = 0
WALL = 1


@dataclass(slots=True)
class MazeTruth:
    h_walls: np.ndarray  # shape: [H+1, W], bool
    v_walls: np.ndarray  # shape: [H, W+1], bool
    blackholes: np.ndarray  # shape: [H, W], bool
    entries: list[tuple[int, int]]


@dataclass(slots=True)
class MapperBelief:
    h_est: np.ndarray  # shape: [H+1, W], int8 in {-1,0,1}
    v_est: np.ndarray  # shape: [H, W+1], int8 in {-1,0,1}
    hole_suspect: np.ndarray  # shape: [H, W], float32 in [0,1]


@dataclass(slots=True)
class WalkerObs:
    pos_idx: int
    local_walls_4bit: int
    anomaly_flag: int
    valid_mask: np.ndarray
    revisit_bin: int
    last_action: int


@dataclass(slots=True)
class MapperObs:
    coverage_bin: int
    stuck_bin: int
    uncertainty_bin: int
    anomaly_bin: int
    entry_dist_bin: int
    reset_budget_bin: int


@dataclass(slots=True)
class ObsPack:
    walker_obs: WalkerObs
    mapper_obs: MapperObs


@dataclass(slots=True)
class RewardPack:
    walker_reward: float
    mapper_reward: float


def _remove_wall_between(h_walls: np.ndarray, v_walls: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        # horizontal move -> vertical wall
        wall_col = max(ac, bc)
        v_walls[ar, wall_col] = False
    else:
        # vertical move -> horizontal wall
        wall_row = max(ar, br)
        h_walls[wall_row, ac] = False


def _neighbors_inside(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ACTION_DELTAS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _open_neighbors_from_walls(
    h_walls: np.ndarray,
    v_walls: np.ndarray,
    h: int,
    w: int,
    pos: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = pos
    out: list[tuple[int, int]] = []
    # up
    if r - 1 >= 0 and not h_walls[r, c]:
        out.append((r - 1, c))
    # right
    if c + 1 < w and not v_walls[r, c + 1]:
        out.append((r, c + 1))
    # down
    if r + 1 < h and not h_walls[r + 1, c]:
        out.append((r + 1, c))
    # left
    if c - 1 >= 0 and not v_walls[r, c]:
        out.append((r, c - 1))
    return out


def _reachable_ratio_from_entry(
    h_walls: np.ndarray,
    v_walls: np.ndarray,
    h: int,
    w: int,
    entry: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> float:
    if entry in blocked:
        return 0.0

    target_cells = h * w - len(blocked)
    if target_cells <= 0:
        return 0.0

    q = deque([entry])
    seen = {entry}
    while q:
        cur = q.popleft()
        for nxt in _open_neighbors_from_walls(h_walls, v_walls, h, w, cur):
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)

    return len(seen) / target_cells


def _can_place_blackhole(
    cfg: MazeConfig,
    h_walls: np.ndarray,
    v_walls: np.ndarray,
    entries: list[tuple[int, int]],
    blocked_existing: set[tuple[int, int]],
    candidate: tuple[int, int],
    min_ratio: float,
) -> bool:
    # Keep holes away from entries so agent is not immediately forced to teleport.
    if min(abs(candidate[0] - er) + abs(candidate[1] - ec) for er, ec in entries) < cfg.blackhole_min_entry_distance:
        return False

    blocked = set(blocked_existing)
    blocked.add(candidate)

    # Each entry should still reach most free cells without stepping on blackholes.
    for entry in entries:
        ratio = _reachable_ratio_from_entry(h_walls, v_walls, cfg.height, cfg.width, entry, blocked)
        if ratio < min_ratio:
            return False
    return True


def place_blackholes(
    cfg: MazeConfig,
    h_walls: np.ndarray,
    v_walls: np.ndarray,
    entries: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = cfg.height, cfg.width
    blackholes = np.zeros((h, w), dtype=bool)
    n_holes = min(cfg.n_blackholes, h * w - len(entries))
    if n_holes <= 0:
        return blackholes

    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in set(entries)]
    rng.shuffle(candidates)

    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()
    for cell in candidates:
        if len(selected) >= n_holes:
            break
        if _can_place_blackhole(
            cfg,
            h_walls,
            v_walls,
            entries,
            blocked_existing=selected_set,
            candidate=cell,
            min_ratio=cfg.blackhole_min_reachable_ratio,
        ):
            selected.append(cell)
            selected_set.add(cell)

    # Relax constraints if strict pass cannot place enough holes.
    if len(selected) < n_holes:
        for cell in candidates:
            if len(selected) >= n_holes:
                break
            if cell in selected_set:
                continue
            if _can_place_blackhole(
                cfg,
                h_walls,
                v_walls,
                entries,
                blocked_existing=selected_set,
                candidate=cell,
                min_ratio=cfg.blackhole_relaxed_reachable_ratio,
            ):
                selected.append(cell)
                selected_set.add(cell)

    for r, c in selected:
        blackholes[r, c] = True
    return blackholes


def generate_connected_maze(cfg: MazeConfig, rng: np.random.Generator) -> MazeTruth:
    h, w = cfg.height, cfg.width
    h_walls = np.ones((h + 1, w), dtype=bool)
    v_walls = np.ones((h, w + 1), dtype=bool)

    # Randomized DFS to produce a connected maze.
    visited = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    stack = [start]
    visited[start] = True

    while stack:
        r, c = stack[-1]
        cand = [(nr, nc) for nr, nc in _neighbors_inside(h, w, r, c) if not visited[nr, nc]]
        if not cand:
            stack.pop()
            continue
        idx = int(rng.integers(0, len(cand)))
        nr, nc = cand[idx]
        _remove_wall_between(h_walls, v_walls, (r, c), (nr, nc))
        visited[nr, nc] = True
        stack.append((nr, nc))

    # Carve additional loops for easier exploration dynamics.
    if cfg.connectivity_carve_extra_prob > 0:
        for r in range(1, h):
            for c in range(w):
                if rng.random() < cfg.connectivity_carve_extra_prob:
                    h_walls[r, c] = False
        for r in range(h):
            for c in range(1, w):
                if rng.random() < cfg.connectivity_carve_extra_prob:
                    v_walls[r, c] = False

    entries = sample_entries(cfg, rng)
    blackholes = place_blackholes(cfg, h_walls, v_walls, entries, rng)

    return MazeTruth(h_walls=h_walls, v_walls=v_walls, blackholes=blackholes, entries=entries)


def sample_entries(cfg: MazeConfig, rng: np.random.Generator) -> list[tuple[int, int]]:
    h, w = cfg.height, cfg.width
    perimeter = []
    for c in range(w):
        perimeter.append((0, c))
        if h > 1:
            perimeter.append((h - 1, c))
    for r in range(1, h - 1):
        perimeter.append((r, 0))
        if w > 1:
            perimeter.append((r, w - 1))

    n = max(1, min(cfg.n_entries, len(perimeter)))
    # Greedy farthest-point sampling to spread entry points.
    first = perimeter[int(rng.integers(0, len(perimeter)))]
    chosen = [first]

    while len(chosen) < n:
        best_cell = None
        best_score = -1
        for cell in perimeter:
            if cell in chosen:
                continue
            score = min(abs(cell[0] - s[0]) + abs(cell[1] - s[1]) for s in chosen)
            if score > best_score:
                best_score = score
                best_cell = cell
        if best_cell is None:
            break
        chosen.append(best_cell)

    return chosen


class SoulgueMazeEnv:
    """Dual-agent tabular-Q environment for maze exploration and reconstruction."""

    def __init__(
        self,
        maze_cfg: MazeConfig,
        reward_cfg: RewardConfig,
        runtime_cfg: EnvRuntimeConfig,
        seed: int = 42,
    ) -> None:
        self.maze_cfg = maze_cfg
        self.reward_cfg = reward_cfg
        self.runtime_cfg = runtime_cfg
        self.rng = np.random.default_rng(seed)

        self.h = maze_cfg.height
        self.w = maze_cfg.width
        self.total_edges = (self.h + 1) * self.w + self.h * (self.w + 1)

        self.truth: MazeTruth | None = None
        self._fixed_truth: MazeTruth | None = None
        self._maze_pool: list[MazeTruth] = []
        self.belief: MapperBelief | None = None
        self.walker_pos = (0, 0)
        self.prev_pos = (0, 0)
        self.recent_positions: Deque[tuple[int, int]] = deque(maxlen=self.runtime_cfg.stuck_window)
        self.visits = np.zeros((self.h, self.w), dtype=np.int32)

        self.step_count = 0
        self.resets_left = self.runtime_cfg.max_resets_per_episode
        self.last_anomaly_flag = 0
        self.last_blackhole_event = False
        self.last_wall_hit = False
        self.last_walker_action = 4
        self.pending_reset: dict | None = None

    def _clone_truth(self, truth: MazeTruth) -> MazeTruth:
        return MazeTruth(
            h_walls=truth.h_walls.copy(),
            v_walls=truth.v_walls.copy(),
            blackholes=truth.blackholes.copy(),
            entries=list(truth.entries),
        )

    def _build_maze_pool(self) -> None:
        if self._maze_pool:
            return
        pool_n = max(0, int(self.maze_cfg.maze_pool_size))
        if pool_n == 0:
            return
        for i in range(pool_n):
            seed = int(self.maze_cfg.maze_pool_seed + i * 9973)
            pool_rng = np.random.default_rng(seed)
            self._maze_pool.append(generate_connected_maze(self.maze_cfg, pool_rng))

    def reset(self, seed: int | None = None) -> ObsPack:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.maze_cfg.fixed_layout_seed is not None:
            if self._fixed_truth is None:
                fixed_rng = np.random.default_rng(self.maze_cfg.fixed_layout_seed)
                self._fixed_truth = generate_connected_maze(self.maze_cfg, fixed_rng)
            self.truth = self._clone_truth(self._fixed_truth)
        else:
            self._build_maze_pool()
            if self._maze_pool:
                idx = int(self.rng.integers(0, len(self._maze_pool)))
                self.truth = self._clone_truth(self._maze_pool[idx])
            else:
                self.truth = generate_connected_maze(self.maze_cfg, self.rng)
        self.belief = MapperBelief(
            h_est=np.full((self.h + 1, self.w), UNKNOWN, dtype=np.int8),
            v_est=np.full((self.h, self.w + 1), UNKNOWN, dtype=np.int8),
            hole_suspect=np.zeros((self.h, self.w), dtype=np.float32),
        )

        start_entry_idx = int(self.rng.integers(0, len(self.truth.entries)))
        self.walker_pos = self.truth.entries[start_entry_idx]
        self.prev_pos = self.walker_pos

        self.visits.fill(0)
        self.visits[self.walker_pos] = 1
        self.recent_positions.clear()
        self.recent_positions.append(self.walker_pos)

        self.step_count = 0
        self.resets_left = self.runtime_cfg.max_resets_per_episode
        self.last_anomaly_flag = 0
        self.last_blackhole_event = False
        self.last_wall_hit = False
        self.last_walker_action = 4
        self.pending_reset = None

        local_walls = self.get_local_walls(self.walker_pos)
        self.update_mapper_belief(self.walker_pos, local_walls)

        return self._build_obs(last_anomaly=0)

    def get_action_mask(self) -> np.ndarray:
        walls = self.get_local_walls(self.walker_pos)
        return np.array([1 - w for w in walls], dtype=np.int8)

    def compute_map_accuracy(self) -> float:
        assert self.truth is not None and self.belief is not None
        h_truth = self.truth.h_walls.astype(np.int8)
        v_truth = self.truth.v_walls.astype(np.int8)

        h_match = self.belief.h_est == h_truth
        v_match = self.belief.v_est == v_truth

        matches = int(h_match.sum() + v_match.sum())
        return matches / self.total_edges

    def compute_coverage(self) -> float:
        assert self.belief is not None
        known = int((self.belief.h_est != UNKNOWN).sum() + (self.belief.v_est != UNKNOWN).sum())
        return known / self.total_edges

    def get_local_walls(self, pos: tuple[int, int]) -> np.ndarray:
        assert self.truth is not None
        r, c = pos
        up = int(self.truth.h_walls[r, c])
        right = int(self.truth.v_walls[r, c + 1])
        down = int(self.truth.h_walls[r + 1, c])
        left = int(self.truth.v_walls[r, c])
        return np.array([up, right, down, left], dtype=np.int8)

    def update_mapper_belief(self, pos: tuple[int, int], local_walls: np.ndarray) -> int:
        assert self.belief is not None
        r, c = pos
        updates = [
            ("h", r, c, int(local_walls[0])),
            ("v", r, c + 1, int(local_walls[1])),
            ("h", r + 1, c, int(local_walls[2])),
            ("v", r, c, int(local_walls[3])),
        ]

        newly_resolved = 0
        for plane, rr, cc, value in updates:
            if plane == "h":
                prev = int(self.belief.h_est[rr, cc])
                if prev == UNKNOWN:
                    newly_resolved += 1
                self.belief.h_est[rr, cc] = value
            else:
                prev = int(self.belief.v_est[rr, cc])
                if prev == UNKNOWN:
                    newly_resolved += 1
                self.belief.v_est[rr, cc] = value
        return newly_resolved

    def _apply_mapper_action(self, mapper_action: int, stuck_bin_before: int) -> tuple[bool, bool]:
        """Apply mapper action. Returns (reset_event, bad_reset)."""
        assert self.truth is not None

        reset_event = False
        bad_reset = False
        if mapper_action in (1, 2) and self.resets_left > 0:
            entry_idx = mapper_action - 1
            if entry_idx < len(self.truth.entries):
                self.prev_pos = self.walker_pos
                self.walker_pos = self.truth.entries[entry_idx]
                self.resets_left -= 1
                reset_event = True
                if stuck_bin_before == 0:
                    bad_reset = True
                self.pending_reset = {
                    "start_accuracy": self.compute_map_accuracy(),
                    "remaining": self.runtime_cfg.reset_recovery_window,
                }
        return reset_event, bad_reset

    def _is_blocked(self, pos: tuple[int, int], action: int) -> bool:
        walls = self.get_local_walls(pos)
        return bool(walls[action] == 1)

    def _move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = ACTION_DELTAS[action]
        return pos[0] + dr, pos[1] + dc

    def _random_cell(self) -> tuple[int, int]:
        r = int(self.rng.integers(0, self.h))
        c = int(self.rng.integers(0, self.w))
        return (r, c)

    def _stuck_bin(self) -> int:
        if len(self.recent_positions) < self.runtime_cfg.stuck_window:
            return 0
        unique_n = len(set(self.recent_positions))
        if unique_n <= self.runtime_cfg.severe_stuck_unique_threshold:
            return 2
        if unique_n <= self.runtime_cfg.mild_stuck_unique_threshold:
            return 1
        return 0

    def _revisit_bin(self, pos: tuple[int, int]) -> int:
        count = int(self.visits[pos])
        if count <= 1:
            return 0
        if count <= 3:
            return 1
        return 2

    def _local_4bit(self, local_walls: np.ndarray) -> int:
        return int(local_walls[0] + 2 * local_walls[1] + 4 * local_walls[2] + 8 * local_walls[3])

    def _entry_dist_bin(self, pos: tuple[int, int]) -> int:
        assert self.truth is not None
        min_dist = min(abs(pos[0] - er) + abs(pos[1] - ec) for er, ec in self.truth.entries)
        max_dist = max(1, self.h + self.w - 2)
        scaled = (min_dist / max_dist) * 4.0
        return min(3, int(scaled))

    def _coverage_bin(self, coverage: float) -> int:
        return min(7, int(coverage * 8.0))

    def _uncertainty_bin(self, accuracy: float) -> int:
        uncertainty = 1.0 - accuracy
        return min(3, int(uncertainty * 4.0))

    def _reset_budget_bin(self) -> int:
        ratio = self.resets_left / max(1, self.runtime_cfg.max_resets_per_episode)
        return min(2, int(ratio * 3.0))

    def _build_obs(self, last_anomaly: int) -> ObsPack:
        local_walls = self.get_local_walls(self.walker_pos)
        valid_mask = np.array([1 - local_walls[0], 1 - local_walls[1], 1 - local_walls[2], 1 - local_walls[3]], dtype=np.int8)
        coverage = self.compute_coverage()
        accuracy = self.compute_map_accuracy()

        walker_obs = WalkerObs(
            pos_idx=self.walker_pos[0] * self.w + self.walker_pos[1],
            local_walls_4bit=self._local_4bit(local_walls),
            anomaly_flag=int(last_anomaly),
            valid_mask=valid_mask,
            revisit_bin=self._revisit_bin(self.walker_pos),
            last_action=self.last_walker_action,
        )

        mapper_obs = MapperObs(
            coverage_bin=self._coverage_bin(coverage),
            stuck_bin=self._stuck_bin(),
            uncertainty_bin=self._uncertainty_bin(accuracy),
            anomaly_bin=int(last_anomaly),
            entry_dist_bin=self._entry_dist_bin(self.walker_pos),
            reset_budget_bin=self._reset_budget_bin(),
        )

        return ObsPack(walker_obs=walker_obs, mapper_obs=mapper_obs)

    def step(self, walker_action: int, mapper_action: int) -> tuple[ObsPack, RewardPack, bool, dict]:
        assert self.truth is not None and self.belief is not None

        self.step_count += 1
        stuck_bin_before = self._stuck_bin()
        accuracy_before = self.compute_map_accuracy()

        reset_event, bad_reset = self._apply_mapper_action(mapper_action, stuck_bin_before)

        wall_hit = False
        blackhole_event = False
        backtrack_event = False

        if not reset_event:
            self.prev_pos = self.walker_pos
            if self._is_blocked(self.walker_pos, walker_action):
                wall_hit = True
            else:
                self.walker_pos = self._move(self.walker_pos, walker_action)
                if self.truth.blackholes[self.walker_pos]:
                    self.walker_pos = self._random_cell()
                    blackhole_event = True
            self.last_walker_action = int(walker_action)
        else:
            self.last_walker_action = 4

        if (not reset_event) and (not wall_hit) and len(self.recent_positions) >= 2:
            # Penalize immediate reversal (A->B->A), a common oscillation failure mode.
            if self.walker_pos == self.recent_positions[-2]:
                backtrack_event = True

        old_visit_count = int(self.visits[self.walker_pos])
        self.visits[self.walker_pos] += 1
        self.recent_positions.append(self.walker_pos)

        delta_dist = abs(self.walker_pos[0] - self.prev_pos[0]) + abs(self.walker_pos[1] - self.prev_pos[1])
        anomaly_flag = int((delta_dist > 1) and (not reset_event))

        if anomaly_flag:
            self.belief.hole_suspect[self.prev_pos] = min(1.0, self.belief.hole_suspect[self.prev_pos] + 0.25)

        local_walls = self.get_local_walls(self.walker_pos)
        newly_resolved = self.update_mapper_belief(self.walker_pos, local_walls)

        accuracy_after = self.compute_map_accuracy()
        coverage_after = self.compute_coverage()
        accuracy_delta = accuracy_after - accuracy_before

        walker_reward = self.reward_cfg.walker_step_cost
        mapper_reward = self.reward_cfg.mapper_step_cost

        if old_visit_count == 0:
            walker_reward += self.reward_cfg.walker_new_cell
        walker_reward += self.reward_cfg.walker_new_edge * newly_resolved

        if wall_hit:
            walker_reward += self.reward_cfg.walker_wall_hit
        if backtrack_event:
            walker_reward += self.reward_cfg.walker_backtrack_penalty
        if self._revisit_bin(self.walker_pos) == 2:
            walker_reward += self.reward_cfg.walker_loop_penalty
        if blackhole_event:
            walker_reward += self.reward_cfg.walker_blackhole_penalty

        mapper_reward += self.reward_cfg.mapper_accuracy_gain_scale * accuracy_delta
        mapper_reward += self.reward_cfg.mapper_new_edge * newly_resolved

        if reset_event:
            mapper_reward += self.reward_cfg.mapper_reset_cost
            if bad_reset:
                mapper_reward += self.reward_cfg.mapper_bad_reset_penalty

        if (mapper_action == 0) and (stuck_bin_before == 2):
            mapper_reward += self.reward_cfg.mapper_noop_stuck_penalty

        if self.pending_reset is not None:
            if accuracy_after > self.pending_reset["start_accuracy"] + 1e-12:
                mapper_reward += self.reward_cfg.mapper_reset_recovery_bonus
                self.pending_reset = None
            else:
                self.pending_reset["remaining"] -= 1
                if self.pending_reset["remaining"] <= 0:
                    self.pending_reset = None

        done = bool(accuracy_after >= self.maze_cfg.accuracy_target)
        if done:
            # Extra team reward for finishing reconstruction in fewer Walker steps.
            fast_finish = max(0.0, 1.0 - (self.step_count / max(1.0, float(self.runtime_cfg.speed_bonus_horizon))))
            speed_bonus = self.reward_cfg.team_fast_finish_bonus_scale * fast_finish
            walker_reward += self.reward_cfg.walker_done_bonus
            mapper_reward += self.reward_cfg.mapper_done_bonus
            walker_reward += speed_bonus
            mapper_reward += speed_bonus

        obs = self._build_obs(last_anomaly=anomaly_flag)
        rewards = RewardPack(walker_reward=float(walker_reward), mapper_reward=float(mapper_reward))

        self.last_anomaly_flag = anomaly_flag
        self.last_blackhole_event = blackhole_event
        self.last_wall_hit = wall_hit

        info = {
            "accuracy": accuracy_after,
            "coverage": coverage_after,
            "step": self.step_count,
            "resets_left": self.resets_left,
            "wall_hit": wall_hit,
            "backtrack_event": backtrack_event,
            "blackhole_event": blackhole_event,
            "anomaly_flag": anomaly_flag,
            "newly_resolved": newly_resolved,
            "reset_event": reset_event,
            "done_reason": "accuracy" if done else "running",
        }
        return obs, rewards, done, info
