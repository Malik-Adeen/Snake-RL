"""Tabular Q-learning agent for Snake.

This module provides an epsilon-greedy tabular Q-learning agent that stores
state-action values in a defaultdict and supports persistence with pickle.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Optional, Tuple, Union


State = Tuple[int, int, int, int, int, int, int, int, int, int, int, int]


class TabularQAgent:
    """Epsilon-greedy tabular Q-learning agent."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self._rng: random.Random = random.Random(seed)
        self.q_table: DefaultDict[State, List[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0]
        )

    def choose_action(self, state: State) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.choice((0, 1, 2))

        q_values: List[float] = self.q_table[state]
        max_q: float = max(q_values)
        best_actions: List[int] = [i for i, q in enumerate(q_values) if q == max_q]
        return self._rng.choice(best_actions)

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        if action not in (0, 1, 2):
            raise ValueError("action must be one of 0, 1, or 2")

        next_max: float = max(self.q_table[next_state])
        target: float = reward + self.gamma * next_max * (1 - int(done))
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Union[str, Path]) -> None:
        save_path = Path(path)
        with save_path.open("wb") as file_obj:
            pickle.dump(dict(self.q_table), file_obj)

    def load(self, path: Union[str, Path]) -> None:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Q-table file not found: {load_path}")

        with load_path.open("rb") as file_obj:
            loaded_q: dict[State, List[float]] = pickle.load(file_obj)

        for state, values in loaded_q.items():
            self.q_table[state] = list(values)

    @property
    def q_table_size(self) -> int:
        return len(self.q_table)


class DoubleQAgent:
    """Double Q-Learning agent with two separate Q-tables.

    Reduces overestimation bias by decoupling action selection from
    action evaluation. Table A selects the greedy action; table B
    evaluates it (and vice versa, chosen randomly each update step).
    Action selection for the policy uses the sum of both tables.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self._rng: random.Random = random.Random(seed)
        self.q_a: DefaultDict[State, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.q_b: DefaultDict[State, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    def choose_action(self, state: State) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.choice((0, 1, 2))

        combined_values: List[float] = [
            self.q_a[state][0] + self.q_b[state][0],
            self.q_a[state][1] + self.q_b[state][1],
            self.q_a[state][2] + self.q_b[state][2],
        ]
        max_q: float = max(combined_values)
        best_actions: List[int] = [i for i, q in enumerate(combined_values) if q == max_q]
        return self._rng.choice(best_actions)

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        if action not in (0, 1, 2):
            raise ValueError("action must be one of 0, 1, or 2")

        if self._rng.random() < 0.5:
            next_values_a: List[float] = self.q_a[next_state]
            max_a: float = max(next_values_a)
            best_a_actions: List[int] = [i for i, q in enumerate(next_values_a) if q == max_a]
            a_star: int = self._rng.choice(best_a_actions)
            target: float = reward + self.gamma * self.q_b[next_state][a_star] * (1 - int(done))
            self.q_a[state][action] += self.alpha * (target - self.q_a[state][action])
        else:
            next_values_b: List[float] = self.q_b[next_state]
            max_b: float = max(next_values_b)
            best_b_actions: List[int] = [i for i, q in enumerate(next_values_b) if q == max_b]
            a_star = self._rng.choice(best_b_actions)
            target = reward + self.gamma * self.q_a[next_state][a_star] * (1 - int(done))
            self.q_b[state][action] += self.alpha * (target - self.q_b[state][action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Union[str, Path]) -> None:
        save_path = Path(path)
        with save_path.open("wb") as file_obj:
            pickle.dump({"q_a": dict(self.q_a), "q_b": dict(self.q_b)}, file_obj)

    def load(self, path: Union[str, Path]) -> None:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Double Q-table file not found: {load_path}")

        with load_path.open("rb") as file_obj:
            loaded: dict[str, dict[State, List[float]]] = pickle.load(file_obj)

        loaded_q_a: dict[State, List[float]] = loaded.get("q_a", {})
        loaded_q_b: dict[State, List[float]] = loaded.get("q_b", {})

        for state, values in loaded_q_a.items():
            self.q_a[state] = list(values)
        for state, values in loaded_q_b.items():
            self.q_b[state] = list(values)

    @property
    def q_table_size(self) -> int:
        return len(set(self.q_a.keys()) | set(self.q_b.keys()))
