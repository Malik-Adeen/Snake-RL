"""Snake game environment for tabular Q-learning.

This module defines a compact, deterministic Snake environment with relative
actions and a small binary state representation suitable for tabular methods.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple


Position = Tuple[int, int]
Observation = Tuple[List[Position], Position]
State = Tuple[int, int, int, int, int, int, int, int, int, int, int, int]


class SnakeEnv:
    """Snake game environment with relative actions."""

    DIRECTIONS: List[Position] = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    ACTION_STRAIGHT: int = 0
    ACTION_LEFT: int = 1
    ACTION_RIGHT: int = 2

    def __init__(self, width: int = 10, height: int = 10, seed: Optional[int] = None) -> None:
        if width < 3 or height < 3:
            raise ValueError("width and height must be at least 3")

        self.width: int = width
        self.height: int = height
        self._rng: random.Random = random.Random(seed)

        self.snake: List[Position] = []
        self.direction_idx: int = 0
        self.food: Position = (0, 0)
        self.score: int = 0
        self.done: bool = False
        self.steps_since_food: int = 0

        self.reset()

    def reset(self) -> Observation:
        center_y: int = self.height // 2
        start_x: int = (self.width - 3) // 2

        self.snake = [
            (start_x + 2, center_y),  # head
            (start_x + 1, center_y),
            (start_x, center_y),
        ]
        self.direction_idx = 0  # facing right
        self.score = 0
        self.done = False
        self.steps_since_food = 0
        self.food = self._spawn_food()
        return (self.snake.copy(), self.food)

    def step(self, action: int) -> Tuple[Observation, float, bool]:
        if self.done:
            raise RuntimeError("Cannot call step() on a finished episode")

        if action not in (self.ACTION_STRAIGHT, self.ACTION_LEFT, self.ACTION_RIGHT):
            raise ValueError("Invalid action")

        if action == self.ACTION_LEFT:
            self.direction_idx = (self.direction_idx - 1) % 4
        elif action == self.ACTION_RIGHT:
            self.direction_idx = (self.direction_idx + 1) % 4

        new_head: Position = self._next_position(self.direction_idx)
        if self._is_collision(new_head):
            self.done = True
            return ((self.snake.copy(), self.food), -10.0, True)

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        old_dist: int = abs(head_x - food_x) + abs(head_y - food_y)

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward: float = 10.0
            self.steps_since_food = 0
            self.food = self._spawn_food()
        else:
            self.snake.pop()
            new_dist: int = abs(new_head[0] - food_x) + abs(new_head[1] - food_y)
            if new_dist < old_dist:
                reward = 0.5
            else:
                reward = -0.5
            self.steps_since_food += 1
            if self.steps_since_food > 0 and self.steps_since_food % 100 == 0:
                additional_penalty: float = min(
                    3.0, 0.5 * float(self.steps_since_food // 100)
                )
                reward -= additional_penalty

        return ((self.snake.copy(), self.food), reward, self.done)

    def get_state(self) -> State:
        """Return 12 binary state features.

        Features:
        danger_straight, danger_left, danger_right,
        food_left, food_right, food_up, food_down,
        food_close, tail_left, tail_right, tail_up, tail_down.
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        straight_idx: int = self.direction_idx
        left_idx: int = (self.direction_idx - 1) % 4
        right_idx: int = (self.direction_idx + 1) % 4

        danger_straight: int = int(self._is_collision(self._next_position(straight_idx)))
        danger_left: int = int(self._is_collision(self._next_position(left_idx)))
        danger_right: int = int(self._is_collision(self._next_position(right_idx)))

        food_left: int = int(food_x < head_x)
        food_right: int = int(food_x > head_x)
        food_up: int = int(food_y < head_y)
        food_down: int = int(food_y > head_y)
        manhattan_dist: int = abs(food_x - head_x) + abs(food_y - head_y)
        food_close: int = int(manhattan_dist <= 3)
        tail_x, tail_y = self.snake[-1]
        tail_left: int = int(tail_x < head_x)
        tail_right: int = int(tail_x > head_x)
        tail_up: int = int(tail_y < head_y)
        tail_down: int = int(tail_y > head_y)

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
        )

    def _is_collision(self, pos: Position) -> bool:
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return pos in self.snake

    def _spawn_food(self) -> Position:
        empty_cells: List[Position] = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.snake
        ]
        if not empty_cells:
            return self.snake[0]
        return self._rng.choice(empty_cells)

    def _next_position(self, direction_idx: int) -> Position:
        dx, dy = self.DIRECTIONS[direction_idx]
        head_x, head_y = self.snake[0]
        return (head_x + dx, head_y + dy)
