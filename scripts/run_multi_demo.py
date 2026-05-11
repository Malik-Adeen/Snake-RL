"""Visual demo for two-snake competitive play."""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import pygame

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import MultiSnakeEnv
from snake_rl.agent import DoubleQAgent
from snake_rl.renderer import MultiSnakeRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-agent Snake visual demo.")
    parser.add_argument(
        "--model_a",
        type=str,
        default="experiments/checkpoints/q_table_multi_a.pkl",
    )
    parser.add_argument("--model_b", type=str, default=None)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_a_path = pathlib.Path(args.model_a)
    if not model_a_path.exists():
        print(f"Error: model file not found at '{model_a_path}'.")
        sys.exit(1)

    agent_a = DoubleQAgent(seed=args.seed)
    agent_a.load(model_a_path)
    agent_a.epsilon = 0.0

    agent_b: DoubleQAgent | None = None
    label_b: str
    if args.model_b is not None:
        model_b_path = pathlib.Path(args.model_b)
        if not model_b_path.exists():
            print(f"Error: model file not found at '{model_b_path}'.")
            sys.exit(1)
        agent_b = DoubleQAgent(seed=args.seed + 1)
        agent_b.load(model_b_path)
        agent_b.epsilon = 0.0
        label_b = "Trained B"
    else:
        label_b = "Random B"

    mode_label = "Trained A vs " + label_b

    env = MultiSnakeEnv(width=args.width, height=args.height, seed=args.seed)
    renderer = MultiSnakeRenderer(grid_width=args.width, grid_height=args.height)
    rng = random.Random(args.seed)
    current_fps: int = args.fps

    try:
        for ep in range(1, args.episodes + 1):
            env.reset()
            done = False
            paused = False
            steps = 0

            while not done and steps < args.max_steps:
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

                if paused:
                    renderer.render(
                        env.snake_a,
                        env.snake_b,
                        env.food,
                        env.score_a,
                        env.score_b,
                        episode=ep,
                        epsilon=args.epsilon,
                        mode_label=mode_label,
                        fps=current_fps,
                    )
                    continue

                if rng.random() < args.epsilon:
                    action_a = rng.choice([0, 1, 2])
                else:
                    action_a = agent_a.choose_action(env.get_state_a())

                if agent_b is None:
                    action_b = rng.choice([0, 1, 2])
                elif rng.random() < args.epsilon:
                    action_b = rng.choice([0, 1, 2])
                else:
                    action_b = agent_b.choose_action(env.get_state_b())

                _, _, _, done_a, done_b, done = env.step(action_a, action_b)
                _ = (done_a, done_b)
                steps += 1

                renderer.render(
                    env.snake_a,
                    env.snake_b,
                    env.food,
                    env.score_a,
                    env.score_b,
                    episode=ep,
                    epsilon=args.epsilon,
                    mode_label=mode_label,
                    fps=current_fps,
                )

            winner = (
                "A"
                if env.score_a > env.score_b
                else "B"
                if env.score_b > env.score_a
                else "Draw"
            )
            print(f"Episode {ep}: A={env.score_a} B={env.score_b} → {winner}")
            pygame.time.wait(1000)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
