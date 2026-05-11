"""Two-snake competitive Snake environment.

This module defines a competitive Snake environment with two agents that move
simultaneously on a shared board, with collision and food rules designed for
adversarial RL training.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple


Position = Tuple[int, int]
State = Tuple[int, ...]
Observation = Tuple[List[Position], List[Position], Position]


class MultiSnakeEnv:
    """Competitive two-snake environment with relative actions."""

    DIRECTIONS: List[Position] = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    ACTION_STRAIGHT: int = 0
    ACTION_LEFT: int = 1
    ACTION_RIGHT: int = 2

    def __init__(self, width: int = 10, height: int = 10, seed: Optional[int] = None) -> None:
        if width < 5 or height < 5:
            raise ValueError("width and height must be at least 5")

        self.width: int = width
        self.height: int = height
        self._rng: random.Random = random.Random(seed)

        self.snake_a: List[Position] = []
        self.snake_b: List[Position] = []
        self.dir_a: int = 0
        self.dir_b: int = 2
        self.score_a: int = 0
        self.score_b: int = 0
        self.done_a: bool = False
        self.done_b: bool = False
        self.food: Position = (0, 0)
        self.done: bool = False

        self.reset()

    def reset(self) -> Observation:
        head_a: Position = (self.width // 4, self.height // 2)
        head_b: Position = (3 * self.width // 4, self.height // 2)

        self.snake_a = [
            head_a,
            (head_a[0] - 1, head_a[1]),
            (head_a[0] - 2, head_a[1]),
        ]
        self.snake_b = [
            head_b,
            (head_b[0] + 1, head_b[1]),
            (head_b[0] + 2, head_b[1]),
        ]

        self.dir_a = 0
        self.dir_b = 2
        self.score_a = 0
        self.score_b = 0
        self.done_a = False
        self.done_b = False
        self.done = False
        self.food = self._spawn_food()

        return (self.snake_a.copy(), self.snake_b.copy(), self.food)

    def step(
        self, action_a: int, action_b: int
    ) -> Tuple[Observation, float, float, bool, bool, bool]:
        if self.done:
            raise RuntimeError("Cannot call step() on a finished episode")

        reward_a: float = 0.0
        reward_b: float = 0.0

        old_food: Position = self.food
        old_dist_a: Optional[int] = None
        old_dist_b: Optional[int] = None

        if not self.done_a:
            self.dir_a = self._update_dir(self.dir_a, action_a)
            head_a = self.snake_a[0]
            old_dist_a = abs(head_a[0] - old_food[0]) + abs(head_a[1] - old_food[1])

        if not self.done_b:
            self.dir_b = self._update_dir(self.dir_b, action_b)
            head_b = self.snake_b[0]
            old_dist_b = abs(head_b[0] - old_food[0]) + abs(head_b[1] - old_food[1])

        new_head_a: Optional[Position] = None
        new_head_b: Optional[Position] = None

        if not self.done_a:
            new_head_a = self._next_pos(self.snake_a[0], self.dir_a)
        if not self.done_b:
            new_head_b = self._next_pos(self.snake_b[0], self.dir_b)

        # 1) Head-on collision.
        if new_head_a is not None and new_head_b is not None and new_head_a == new_head_b:
            self.done_a = True
            self.done_b = True
            reward_a = -10.0
            reward_b = -10.0

        # 2) A collision checks.
        if not self.done_a and new_head_a is not None and self._is_collision_for_a(new_head_a):
            self.done_a = True
            reward_a = -10.0

        # 3) B collision checks.
        if not self.done_b and new_head_b is not None and self._is_collision_for_b(new_head_b):
            self.done_b = True
            reward_b = -10.0

        ate_a: bool = False
        ate_b: bool = False

        if not self.done_a and new_head_a is not None:
            self.snake_a.insert(0, new_head_a)
            if new_head_a == old_food:
                ate_a = True
                self.score_a += 1
                reward_a = 10.0
                self.food = self._spawn_food()
            else:
                self.snake_a.pop()

        if not self.done_b and new_head_b is not None:
            self.snake_b.insert(0, new_head_b)
            if (not ate_a) and (new_head_b == old_food):
                ate_b = True
                self.score_b += 1
                reward_b = 10.0
                self.food = self._spawn_food()
            else:
                self.snake_b.pop()

        # 5) Default step reward based on distance to food before movement.
        if not self.done_a and (not ate_a) and new_head_a is not None and old_dist_a is not None:
            new_dist_a: int = abs(new_head_a[0] - old_food[0]) + abs(new_head_a[1] - old_food[1])
            reward_a = 0.5 if new_dist_a < old_dist_a else -0.5

        if not self.done_b and (not ate_b) and new_head_b is not None and old_dist_b is not None:
            new_dist_b: int = abs(new_head_b[0] - old_food[0]) + abs(new_head_b[1] - old_food[1])
            reward_b = 0.5 if new_dist_b < old_dist_b else -0.5

        self.done = self.done_a and self.done_b

        obs: Observation = (self.snake_a.copy(), self.snake_b.copy(), self.food)
        return (obs, reward_a, reward_b, self.done_a, self.done_b, self.done)

    def get_state_a(self) -> State:
        if self.done_a or not self.snake_a:
            return tuple(0 for _ in range(16))
        return self._get_state("a")

    def get_state_b(self) -> State:
        if self.done_b or not self.snake_b:
            return tuple(0 for _ in range(16))
        return self._get_state("b")

    def _get_state(self, agent: str) -> State:
        if agent == "a":
            own_snake: List[Position] = self.snake_a
            own_dir: int = self.dir_a
            opp_snake: List[Position] = self.snake_b
            collision_fn = self._is_collision_for_a
        else:
            own_snake = self.snake_b
            own_dir = self.dir_b
            opp_snake = self.snake_a
            collision_fn = self._is_collision_for_b

        head_x, head_y = own_snake[0]
        food_x, food_y = self.food

        straight_idx: int = own_dir
        left_idx: int = (own_dir - 1) % 4
        right_idx: int = (own_dir + 1) % 4

        danger_straight: int = int(collision_fn(self._next_pos(own_snake[0], straight_idx)))
        danger_left: int = int(collision_fn(self._next_pos(own_snake[0], left_idx)))
        danger_right: int = int(collision_fn(self._next_pos(own_snake[0], right_idx)))

        food_left: int = int(food_x < head_x)
        food_right: int = int(food_x > head_x)
        food_up: int = int(food_y < head_y)
        food_down: int = int(food_y > head_y)
        food_close: int = int(abs(food_x - head_x) + abs(food_y - head_y) <= 3)

        tail_x, tail_y = own_snake[-1]
        tail_left: int = int(tail_x < head_x)
        tail_right: int = int(tail_x > head_x)
        tail_up: int = int(tail_y < head_y)
        tail_down: int = int(tail_y > head_y)

        if opp_snake:
            opp_head_x, opp_head_y = opp_snake[0]
        else:
            opp_head_x, opp_head_y = head_x, head_y
        opp_left: int = int(opp_head_x < head_x)
        opp_right: int = int(opp_head_x > head_x)
        opp_up: int = int(opp_head_y < head_y)
        opp_down: int = int(opp_head_y > head_y)

        return (
            danger_straight,
            danger_left,
            danger_right,
            food_left,
            food_right,
            food_up,
            food_down,
            food_close,
            tail_left,
            tail_right,
            tail_up,
            tail_down,
            opp_left,
            opp_right,
            opp_up,
            opp_down,
        )

    def _is_collision_for_a(self, pos: Position) -> bool:
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return pos in self.snake_a or pos in self.snake_b

    def _is_collision_for_b(self, pos: Position) -> bool:
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return pos in self.snake_b or pos in self.snake_a

    def _spawn_food(self) -> Position:
        occupied: Dict[Position, bool] = {segment: True for segment in self.snake_a}
        occupied.update({segment: True for segment in self.snake_b})
        empty_cells: List[Position] = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not empty_cells:
            if self.snake_a:
                return self.snake_a[0]
            if self.snake_b:
                return self.snake_b[0]
            return (0, 0)
        return self._rng.choice(empty_cells)

    def _next_pos(self, head: Position, dir_idx: int) -> Position:
        dx, dy = self.DIRECTIONS[dir_idx]
        return (head[0] + dx, head[1] + dy)

    def _update_dir(self, dir_idx: int, action: int) -> int:
        if action not in (self.ACTION_STRAIGHT, self.ACTION_LEFT, self.ACTION_RIGHT):
            raise ValueError("Invalid action")
        if action == self.ACTION_LEFT:
            return (dir_idx - 1) % 4
        if action == self.ACTION_RIGHT:
            return (dir_idx + 1) % 4
        return dir_idx
