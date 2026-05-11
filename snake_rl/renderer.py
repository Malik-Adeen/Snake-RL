"""Pygame renderer for visualizing Snake RL episodes.

This module provides rendering utilities for single-agent and multi-agent Snake
episodes, with compact HUD overlays and stylized game entities.
"""

from __future__ import annotations

from math import sin
from typing import Sequence, Tuple

import pygame

Position = Tuple[int, int]
Color = Tuple[int, int, int]


def _lerp_color(start: Color, end: Color, t: float) -> Color:
    clamped_t = max(0.0, min(1.0, t))
    return (
        int(start[0] + (end[0] - start[0]) * clamped_t),
        int(start[1] + (end[1] - start[1]) * clamped_t),
        int(start[2] + (end[2] - start[2]) * clamped_t),
    )


def _darker_color(color: Color, factor: float = 0.6) -> Color:
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def _snake_direction(snake: Sequence[Position], default_dir: Position = (1, 0)) -> Position:
    if len(snake) <= 1:
        return default_dir
    head_x, head_y = snake[0]
    neck_x, neck_y = snake[1]
    return (head_x - neck_x, head_y - neck_y)


def _cell_center(cell: Position, cell_size: int) -> Position:
    return (
        cell[0] * cell_size + cell_size // 2,
        cell[1] * cell_size + cell_size // 2,
    )


def _draw_food(surface: pygame.Surface, food: Position, cell_size: int, food_color: Color) -> None:
    ticks = pygame.time.get_ticks()
    base_radius: int = max(3, cell_size // 3)
    pulse: float = sin(ticks / 400.0) * float(max(1, cell_size // 10))
    radius: int = max(3, int(base_radius + pulse))

    food_center = _cell_center(food, cell_size)
    pygame.draw.circle(surface, food_color, food_center, radius)

    highlight_radius: int = max(1, base_radius // 3)
    highlight_center = (
        food_center[0] - radius // 3,
        food_center[1] - radius // 3,
    )
    pygame.draw.circle(surface, (255, 255, 255), highlight_center, highlight_radius)


def _draw_eyes(
    surface: pygame.Surface,
    head: Position,
    direction: Position,
    cell_size: int,
) -> None:
    center_x, center_y = _cell_center(head, cell_size)
    eye_offset: int = max(2, cell_size // 4)
    eye_sep: int = max(2, cell_size // 6)
    eye_radius: int = max(2, cell_size // 8)
    pupil_radius: int = max(1, cell_size // 12)

    if direction == (1, 0):  # right
        eyes = [
            (center_x + eye_offset, center_y - eye_sep),
            (center_x + eye_offset, center_y + eye_sep),
        ]
    elif direction == (-1, 0):  # left
        eyes = [
            (center_x - eye_offset, center_y - eye_sep),
            (center_x - eye_offset, center_y + eye_sep),
        ]
    elif direction == (0, -1):  # up
        eyes = [
            (center_x - eye_sep, center_y - eye_offset),
            (center_x + eye_sep, center_y - eye_offset),
        ]
    else:  # down
        eyes = [
            (center_x - eye_sep, center_y + eye_offset),
            (center_x + eye_sep, center_y + eye_offset),
        ]

    for ex, ey in eyes:
        pygame.draw.circle(surface, (255, 255, 255), (ex, ey), eye_radius)
        pygame.draw.circle(surface, (0, 0, 0), (ex, ey), pupil_radius)


def _draw_snake_body_and_head(
    surface: pygame.Surface,
    snake: Sequence[Position],
    cell_size: int,
    body_color: Color,
    head_color: Color,
    default_dir: Position = (1, 0),
) -> None:
    if not snake:
        return

    margin: int = 3
    inner_size: int = max(1, cell_size - 2 * margin)
    tail_color: Color = _darker_color(body_color, 0.6)
    denom: int = max(len(snake) - 1, 1)

    segment_colors = [head_color]
    for index in range(1, len(snake)):
        t = float(index) / float(denom)
        segment_colors.append(_lerp_color(body_color, tail_color, t))

    for idx in range(len(snake) - 1):
        first = snake[idx]
        second = snake[idx + 1]
        c1x, c1y = _cell_center(first, cell_size)
        c2x, c2y = _cell_center(second, cell_size)

        if c1x != c2x:  # horizontal bridge
            left = min(c1x, c2x)
            bridge_rect = pygame.Rect(left, c1y - inner_size // 2, abs(c2x - c1x), inner_size)
        else:  # vertical bridge
            top = min(c1y, c2y)
            bridge_rect = pygame.Rect(c1x - inner_size // 2, top, inner_size, abs(c2y - c1y))
        pygame.draw.rect(surface, segment_colors[idx + 1], bridge_rect)

    for idx, (seg_x, seg_y) in enumerate(snake[1:], start=1):
        seg_rect = pygame.Rect(
            seg_x * cell_size + margin,
            seg_y * cell_size + margin,
            inner_size,
            inner_size,
        )
        pygame.draw.rect(
            surface,
            segment_colors[idx],
            seg_rect,
            border_radius=max(2, cell_size // 3),
        )

    head_x, head_y = snake[0]
    head_rect = pygame.Rect(
        head_x * cell_size + margin,
        head_y * cell_size + margin,
        inner_size,
        inner_size,
    )
    pygame.draw.rect(
        surface,
        head_color,
        head_rect,
        border_radius=max(2, cell_size // 2),
    )

    _draw_eyes(surface, snake[0], _snake_direction(snake, default_dir=default_dir), cell_size)


class SnakeRenderer:
    BG = (15, 15, 20)
    GRID = (30, 30, 40)
    SNAKE_BODY = (34, 180, 100)
    SNAKE_HEAD = (80, 255, 140)
    FOOD = (220, 60, 60)
    HUD_BG = (25, 25, 35)
    TEXT = (220, 220, 220)
    TEXT_DIM = (120, 120, 140)

    def __init__(self, grid_width: int = 10, grid_height: int = 10, cell_size: int = 40) -> None:
        self.grid_width: int = grid_width
        self.grid_height: int = grid_height
        self.cell_size: int = cell_size
        self.hud_height: int = 80
        self.window_width: int = self.grid_width * self.cell_size
        self.window_height: int = self.grid_height * self.cell_size + self.hud_height

        pygame.init()
        self.screen: pygame.Surface = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption("Snake RL")
        self.font: pygame.font.Font = pygame.font.SysFont("monospace", 18)
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def _draw_grid(self, surface: pygame.Surface) -> int:
        grid_pixel_height: int = self.grid_height * self.cell_size
        surface.fill(self.BG)
        for x in range(self.grid_width + 1):
            px = x * self.cell_size
            pygame.draw.line(surface, self.GRID, (px, 0), (px, grid_pixel_height), 1)
        for y in range(self.grid_height + 1):
            py = y * self.cell_size
            pygame.draw.line(surface, self.GRID, (0, py), (self.window_width, py), 1)
        return grid_pixel_height

    def _draw_hud(
        self,
        surface: pygame.Surface,
        grid_pixel_height: int,
        score: int,
        episode: int,
        epsilon: float,
        agent_label: str,
    ) -> None:
        hud_rect = pygame.Rect(0, grid_pixel_height, self.window_width, self.hud_height)
        pygame.draw.rect(surface, self.HUD_BG, hud_rect)
        pygame.draw.line(surface, (50, 50, 65), (0, grid_pixel_height), (self.window_width, grid_pixel_height), 1)

        left_line_1 = self.font.render(f"Episode: {episode}", True, self.TEXT)
        left_line_2 = self.font.render(f"Score: {score}", True, self.TEXT)
        right_line_1 = self.font.render(f"ε: {epsilon:.3f}", True, self.TEXT)
        right_line_2 = self.font.render(f"Agent: {agent_label}", True, self.TEXT_DIM)

        surface.blit(left_line_1, (12, grid_pixel_height + 12))
        surface.blit(left_line_2, (12, grid_pixel_height + 40))
        surface.blit(
            right_line_1,
            (self.window_width - right_line_1.get_width() - 12, grid_pixel_height + 12),
        )
        surface.blit(
            right_line_2,
            (self.window_width - right_line_2.get_width() - 12, grid_pixel_height + 40),
        )

    def render_to_surface(
        self,
        surface: pygame.Surface,
        snake: Sequence[Position],
        food: Position,
        score: int,
        episode: int,
        epsilon: float,
        agent_label: str,
    ) -> None:
        grid_pixel_height = self._draw_grid(surface)
        _draw_snake_body_and_head(
            surface,
            snake,
            self.cell_size,
            self.SNAKE_BODY,
            self.SNAKE_HEAD,
            default_dir=(1, 0),
        )
        _draw_food(surface, food, self.cell_size, self.FOOD)
        self._draw_hud(surface, grid_pixel_height, score, episode, epsilon, agent_label)

    def render(
        self,
        snake: Sequence[Position],
        food: Position,
        score: int,
        episode: int,
        epsilon: float,
        agent_label: str,
        fps: int = 15,
    ) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                raise SystemExit

        self.render_to_surface(
            self.screen,
            snake,
            food,
            score,
            episode,
            epsilon,
            agent_label,
        )
        pygame.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()


class MultiSnakeRenderer:
    SNAKE_A_BODY = (34, 180, 100)
    SNAKE_A_HEAD = (80, 255, 140)
    SNAKE_B_BODY = (65, 105, 225)
    SNAKE_B_HEAD = (100, 180, 255)
    FOOD = (220, 60, 60)
    BG = (15, 15, 20)
    GRID = (30, 30, 40)
    HUD_BG = (25, 25, 35)
    TEXT = (220, 220, 220)
    TEXT_DIM = (120, 120, 140)

    def __init__(self, grid_width: int = 10, grid_height: int = 10, cell_size: int = 40) -> None:
        self.grid_width: int = grid_width
        self.grid_height: int = grid_height
        self.cell_size: int = cell_size
        self.hud_height: int = 80
        self.window_width: int = self.grid_width * self.cell_size
        self.window_height: int = self.grid_height * self.cell_size + self.hud_height

        pygame.init()
        self.screen: pygame.Surface = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption("Snake RL — Multi Agent")
        self.font: pygame.font.Font = pygame.font.SysFont("monospace", 18)
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def render(
        self,
        snake_a: Sequence[Position],
        snake_b: Sequence[Position],
        food: Position,
        score_a: int,
        score_b: int,
        episode: int,
        epsilon: float,
        mode_label: str,
        fps: int = 15,
    ) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                raise SystemExit

        grid_pixel_height: int = self.grid_height * self.cell_size
        self.screen.fill(self.BG)

        for x in range(self.grid_width + 1):
            px: int = x * self.cell_size
            pygame.draw.line(self.screen, self.GRID, (px, 0), (px, grid_pixel_height), 1)
        for y in range(self.grid_height + 1):
            py: int = y * self.cell_size
            pygame.draw.line(self.screen, self.GRID, (0, py), (self.window_width, py), 1)

        _draw_snake_body_and_head(
            self.screen,
            snake_b,
            self.cell_size,
            self.SNAKE_B_BODY,
            self.SNAKE_B_HEAD,
            default_dir=(-1, 0),
        )
        _draw_snake_body_and_head(
            self.screen,
            snake_a,
            self.cell_size,
            self.SNAKE_A_BODY,
            self.SNAKE_A_HEAD,
            default_dir=(1, 0),
        )
        _draw_food(self.screen, food, self.cell_size, self.FOOD)

        hud_rect = pygame.Rect(0, grid_pixel_height, self.window_width, self.hud_height)
        pygame.draw.rect(self.screen, self.HUD_BG, hud_rect)
        pygame.draw.line(
            self.screen,
            (50, 50, 65),
            (0, grid_pixel_height),
            (self.window_width, grid_pixel_height),
            1,
        )

        left_line_1 = self.font.render(f"Episode: {episode}", True, self.TEXT)
        left_line_2 = self.font.render(f"A: {score_a}  B: {score_b}", True, self.TEXT)
        right_line_1 = self.font.render(f"ε: {epsilon:.3f}", True, self.TEXT)
        right_line_2 = self.font.render(mode_label, True, self.TEXT_DIM)

        self.screen.blit(left_line_1, (12, grid_pixel_height + 12))
        self.screen.blit(left_line_2, (12, grid_pixel_height + 40))
        self.screen.blit(
            right_line_1,
            (self.window_width - right_line_1.get_width() - 12, grid_pixel_height + 12),
        )
        self.screen.blit(
            right_line_2,
            (self.window_width - right_line_2.get_width() - 12, grid_pixel_height + 40),
        )

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
