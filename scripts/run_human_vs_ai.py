"""Run a human-vs-AI multi-agent Snake demo."""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Optional

import pygame

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import MultiSnakeEnv
from snake_rl.agent import DoubleQAgent
from snake_rl.renderer import MultiSnakeRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Snake B against trained AI Snake A.")
    parser.add_argument(
        "--model_a",
        type=str,
        default="experiments/checkpoints/q_table_multi_a.pkl",
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _absolute_to_relative_action(env: MultiSnakeEnv, target_dir: Optional[int]) -> int:
    if target_dir is None:
        return env.ACTION_STRAIGHT
    current_dir = env.dir_b
    if target_dir == current_dir:
        return env.ACTION_STRAIGHT
    if target_dir == (current_dir + 2) % 4:
        return env.ACTION_STRAIGHT
    if target_dir == (current_dir - 1) % 4:
        return env.ACTION_LEFT
    if target_dir == (current_dir + 1) % 4:
        return env.ACTION_RIGHT
    return env.ACTION_STRAIGHT


def _draw_human_hud(renderer: MultiSnakeRenderer, episode: int, score_ai: int, score_you: int) -> None:
    grid_pixel_height = renderer.grid_height * renderer.cell_size
    pygame.draw.rect(
        renderer.screen,
        renderer.HUD_BG,
        pygame.Rect(0, grid_pixel_height, renderer.window_width, renderer.hud_height),
    )
    pygame.draw.line(
        renderer.screen,
        (50, 50, 65),
        (0, grid_pixel_height),
        (renderer.window_width, grid_pixel_height),
        1,
    )

    font_label = pygame.font.SysFont("monospace", 20)
    font_info = pygame.font.SysFont("monospace", 18)

    ai_label = font_label.render("AI", True, renderer.SNAKE_A_HEAD)
    ai_score = font_info.render(f"{score_ai}", True, renderer.SNAKE_A_HEAD)
    you_label = font_label.render("YOU", True, renderer.SNAKE_B_HEAD)
    you_score = font_info.render(f"{score_you}", True, renderer.SNAKE_B_HEAD)
    ep_text = font_info.render(f"Episode: {episode}", True, renderer.TEXT)

    renderer.screen.blit(ai_label, (12, grid_pixel_height + 10))
    renderer.screen.blit(ai_score, (12, grid_pixel_height + 42))
    renderer.screen.blit(
        ep_text,
        (renderer.window_width // 2 - ep_text.get_width() // 2, grid_pixel_height + 25),
    )
    renderer.screen.blit(
        you_label,
        (renderer.window_width - you_label.get_width() - 12, grid_pixel_height + 10),
    )
    renderer.screen.blit(
        you_score,
        (renderer.window_width - you_score.get_width() - 12, grid_pixel_height + 42),
    )


def _draw_center_message(renderer: MultiSnakeRenderer, text: str, color: tuple[int, int, int]) -> None:
    font = pygame.font.SysFont("monospace", 52)
    message = font.render(text, True, color)
    board_width = renderer.grid_width * renderer.cell_size
    board_height = renderer.grid_height * renderer.cell_size
    renderer.screen.blit(
        message,
        (
            board_width // 2 - message.get_width() // 2,
            board_height // 2 - message.get_height() // 2,
        ),
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    model_path = pathlib.Path(args.model_a)
    if not model_path.exists():
        print(f"Error: model file not found at '{model_path}'.")
        sys.exit(1)

    agent_a = DoubleQAgent(seed=args.seed)
    agent_a.load(model_path)
    agent_a.epsilon = 0.0

    env = MultiSnakeEnv(width=args.width, height=args.height, seed=args.seed)
    renderer = MultiSnakeRenderer(grid_width=args.width, grid_height=args.height)
    current_fps = args.fps

    key_to_dir = {
        pygame.K_RIGHT: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_UP: 3,
    }

    try:
        for ep in range(1, args.episodes + 1):
            env.reset()

            for text, wait_ms in (("3...", 1000), ("2...", 1000), ("1...", 1000), ("GO!", 800)):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                        current_fps = min(60, current_fps + 5)
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                        current_fps = max(1, current_fps - 5)

                renderer.render(
                    env.snake_a,
                    env.snake_b,
                    env.food,
                    env.score_a,
                    env.score_b,
                    episode=ep,
                    epsilon=args.epsilon,
                    mode_label="AI vs YOU",
                    fps=current_fps,
                )
                _draw_human_hud(renderer, episode=ep, score_ai=env.score_a, score_you=env.score_b)
                _draw_center_message(renderer, text, (240, 240, 240))
                pygame.display.flip()
                pygame.time.wait(wait_ms)

            while not env.done:
                desired_dir: Optional[int] = None
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key in key_to_dir:
                        desired_dir = key_to_dir[event.key]
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_EQUALS:
                        current_fps = min(60, current_fps + 2)
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_MINUS:
                        current_fps = max(1, current_fps - 2)

                if rng.random() < args.epsilon:
                    action_a = rng.choice([0, 1, 2])
                else:
                    action_a = agent_a.choose_action(env.get_state_a())

                action_b = _absolute_to_relative_action(env, desired_dir)
                _, _, _, _, _, _ = env.step(action_a, action_b)

                renderer.render(
                    env.snake_a,
                    env.snake_b,
                    env.food,
                    env.score_a,
                    env.score_b,
                    episode=ep,
                    epsilon=args.epsilon,
                    mode_label="AI vs YOU",
                    fps=current_fps,
                )
                _draw_human_hud(renderer, episode=ep, score_ai=env.score_a, score_you=env.score_b)
                pygame.display.flip()

            if env.score_b > env.score_a:
                result_text = "YOU WIN!"
                result_color = renderer.SNAKE_B_HEAD
                result_label = "YOU WIN"
            elif env.score_a > env.score_b:
                result_text = "AI WINS!"
                result_color = renderer.SNAKE_A_HEAD
                result_label = "AI WINS"
            else:
                result_text = "DRAW!"
                result_color = (240, 240, 240)
                result_label = "DRAW"

            print(f"Episode {ep} | AI: {env.score_a} | You: {env.score_b} | Result: {result_label}")

            renderer.render(
                env.snake_a,
                env.snake_b,
                env.food,
                env.score_a,
                env.score_b,
                episode=ep,
                epsilon=args.epsilon,
                mode_label="AI vs YOU",
                fps=current_fps,
            )
            _draw_human_hud(renderer, episode=ep, score_ai=env.score_a, score_you=env.score_b)
            _draw_center_message(renderer, result_text, result_color)
            pygame.display.flip()
            pygame.time.wait(2000)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
