"""Tabular Q-learning agents for Walker and Mapper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env import N_MAPPER_ACTIONS, N_WALKER_ACTIONS, MapperObs, WalkerObs


def q_update(
    q_table: np.ndarray,
    s: int,
    a: int,
    r: float,
    s_next: int,
    alpha: float,
    gamma: float,
    done: bool,
) -> None:
    target = r if done else r + gamma * float(np.max(q_table[s_next]))
    q_table[s, a] += alpha * (target - q_table[s, a])


@dataclass(slots=True)
class WalkerQAgent:
    height: int
    width: int
    alpha: float
    gamma: float
    n_states: int = field(init=False)
    n_actions: int = field(init=False)
    q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # pos*walls(16)*anomaly(2)*revisit(3)*last_action(5)
        self.n_states = self.height * self.width * 16 * 2 * 3 * 5
        self.n_actions = N_WALKER_ACTIONS
        self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def encode_state(self, obs: WalkerObs) -> int:
        pos_block = obs.pos_idx
        state_id = pos_block
        state_id = state_id * 16 + int(obs.local_walls_4bit)
        state_id = state_id * 2 + int(obs.anomaly_flag)
        state_id = state_id * 3 + int(obs.revisit_bin)
        state_id = state_id * 5 + int(obs.last_action)
        return state_id

    def select_action(self, state_id: int, valid_mask: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        valid_actions = np.where(valid_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0

        if rng.random() < epsilon:
            idx = int(rng.integers(0, len(valid_actions)))
            return int(valid_actions[idx])

        q_row = self.q[state_id].copy()
        invalid_actions = np.where(valid_mask <= 0)[0]
        q_row[invalid_actions] = -1e9
        best = np.flatnonzero(q_row == np.max(q_row))
        return int(best[int(rng.integers(0, len(best)))])

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        q_update(self.q, s, a, r, s_next, self.alpha, self.gamma, done)


@dataclass(slots=True)
class MapperQAgent:
    alpha: float
    gamma: float
    reset_margin: float = 0.2
    n_states: int = field(init=False)
    n_actions: int = field(init=False)
    q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # coverage(8)*stuck(3)*uncertainty(4)*anomaly(2)*entry_dist(4)*reset_budget(3)
        self.n_states = 8 * 3 * 4 * 2 * 4 * 3
        self.n_actions = N_MAPPER_ACTIONS
        self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def encode_state(self, obs: MapperObs) -> int:
        state_id = obs.coverage_bin
        state_id = state_id * 3 + obs.stuck_bin
        state_id = state_id * 4 + obs.uncertainty_bin
        state_id = state_id * 2 + obs.anomaly_bin
        state_id = state_id * 4 + obs.entry_dist_bin
        state_id = state_id * 3 + obs.reset_budget_bin
        return state_id

    def select_action(self, state_id: int, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            # Biased exploration: avoid overusing resets early in training.
            return int(rng.choice(np.array([0, 0, 0, 1, 2], dtype=np.int8)))
        q_row = self.q[state_id]
        if float(np.max(q_row)) - float(q_row[0]) <= self.reset_margin:
            return 0
        best = np.flatnonzero(q_row == np.max(q_row))
        return int(best[int(rng.integers(0, len(best)))])

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        q_update(self.q, s, a, r, s_next, self.alpha, self.gamma, done)
