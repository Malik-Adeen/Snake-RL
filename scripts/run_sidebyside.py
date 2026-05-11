"""Run random vs trained Snake side-by-side in one window."""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Tuple

import pygame

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import SnakeEnv, TabularQAgent
from snake_rl.agent import DoubleQAgent
from snake_rl.renderer import SnakeRenderer

State = Tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random vs trained Snake side-by-side.")
    parser.add_argument("--model", type=str, default="experiments/checkpoints/q_table_double.pkl")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build_panel_renderer(
    surface: pygame.Surface,
    grid_width: int,
    grid_height: int,
    cell_size: int,
) -> SnakeRenderer:
    renderer = SnakeRenderer.__new__(SnakeRenderer)
    renderer.grid_width = grid_width
    renderer.grid_height = grid_height
    renderer.cell_size = cell_size
    renderer.hud_height = 80
    renderer.window_width = grid_width * cell_size
    renderer.window_height = grid_height * cell_size + renderer.hud_height
    renderer.screen = surface
    renderer.font = pygame.font.SysFont("monospace", 18)
    renderer.clock = pygame.time.Clock()
    return renderer


def _draw_wide_hud(
    screen: pygame.Surface,
    panel_w: int,
    grid_h_px: int,
    window_w: int,
    score_random: int,
    score_trained: int,
    episode: int,
    epsilon: float,
) -> None:
    hud_rect = pygame.Rect(0, grid_h_px, window_w, 80)
    pygame.draw.rect(screen, SnakeRenderer.HUD_BG, hud_rect)
    pygame.draw.line(screen, (50, 50, 65), (0, grid_h_px), (window_w, grid_h_px), 1)

    font_main = pygame.font.SysFont("monospace", 20)
    font_sub = pygame.font.SysFont("monospace", 18)

    random_label = font_main.render("RANDOM", True, (220, 120, 120))
    random_score = font_sub.render(f"Score: {score_random}", True, (220, 120, 120))
    trained_label = font_main.render("TRAINED", True, (120, 220, 120))
    trained_score = font_sub.render(f"Score: {score_trained}", True, (120, 220, 120))
    episode_text = font_sub.render(f"Episode: {episode}", True, SnakeRenderer.TEXT)
    epsilon_text = font_sub.render(f"ε: {epsilon:.3f}", True, SnakeRenderer.TEXT)

    quarter = window_w // 4
    screen.blit(random_label, (quarter // 2 - random_label.get_width() // 2, grid_h_px + 8))
    screen.blit(random_score, (quarter // 2 - random_score.get_width() // 2, grid_h_px + 40))
    screen.blit(episode_text, (quarter + quarter // 2 - episode_text.get_width() // 2, grid_h_px + 25))
    screen.blit(
        epsilon_text,
        (2 * quarter + quarter // 2 - epsilon_text.get_width() // 2, grid_h_px + 25),
    )
    screen.blit(
        trained_label,
        (3 * quarter + quarter // 2 - trained_label.get_width() // 2, grid_h_px + 8),
    )
    screen.blit(
        trained_score,
        (3 * quarter + quarter // 2 - trained_score.get_width() // 2, grid_h_px + 40),
    )

    divider_x = panel_w + 6
    pygame.draw.line(screen, (255, 255, 255), (divider_x, 0), (divider_x, grid_h_px + 80), 3)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found at '{model_path}'.")
        sys.exit(1)

    trained_agent = DoubleQAgent(seed=args.seed) if args.double else TabularQAgent(seed=args.seed)
    trained_agent.load(model_path)
    trained_agent.epsilon = 0.0

    cell_size = 40
    panel_w = args.width * cell_size
    panel_h = args.height * cell_size + 80
    window_w = 2 * panel_w + 12
    window_h = panel_h

    pygame.init()
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("Snake RL — Side by Side")
    left_surface = screen.subsurface((0, 0, panel_w, panel_h))
    right_surface = screen.subsurface((panel_w + 12, 0, panel_w, panel_h))
    left_renderer = _build_panel_renderer(left_surface, args.width, args.height, cell_size)
    right_renderer = _build_panel_renderer(right_surface, args.width, args.height, cell_size)
    clock = pygame.time.Clock()
    current_fps = args.fps

    try:
        for ep in range(1, args.episodes + 1):
            episode_seed = args.seed + ep
            env_random = SnakeEnv(width=args.width, height=args.height, seed=episode_seed)
            env_trained = SnakeEnv(width=args.width, height=args.height, seed=episode_seed)
            env_random.reset()
            env_trained.reset()

            done_random = False
            done_trained = False
            paused = False
            steps = 0

            while (not done_random or not done_trained) and steps < args.max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                        current_fps = min(60, current_fps + 5)
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                        current_fps = max(1, current_fps - 5)

                if not paused:
                    if not done_random:
                        action_random = rng.choice([0, 1, 2])
                        _, _, done_random = env_random.step(action_random)

                    if not done_trained:
                        state: State = env_trained.get_state()
                        if rng.random() < args.epsilon:
                            action_trained = rng.choice([0, 1, 2])
                        else:
                            action_trained = trained_agent.choose_action(state)
                        _, _, done_trained = env_trained.step(action_trained)

                    steps += 1

                left_renderer.render_to_surface(
                    left_surface,
                    env_random.snake,
                    env_random.food,
                    env_random.score,
                    ep,
                    args.epsilon,
                    "Random",
                )
                right_renderer.render_to_surface(
                    right_surface,
                    env_trained.snake,
                    env_trained.food,
                    env_trained.score,
                    ep,
                    args.epsilon,
                    "Trained",
                )
                _draw_wide_hud(
                    screen,
                    panel_w=panel_w,
                    grid_h_px=args.height * cell_size,
                    window_w=window_w,
                    score_random=env_random.score,
                    score_trained=env_trained.score,
                    episode=ep,
                    epsilon=args.epsilon,
                )
                pygame.display.flip()
                clock.tick(current_fps)

            winner = (
                "Random"
                if env_random.score > env_trained.score
                else "Trained"
                if env_trained.score > env_random.score
                else "Draw"
            )
            print(
                f"Episode {ep} | Random: {env_random.score} | "
                f"Trained: {env_trained.score} | Winner: {winner}"
            )
            pygame.time.wait(700)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
