import argparse
import pathlib
import sys

import pygame

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from snake_rl import SnakeEnv, SnakeRenderer
from snake_rl.dqn_agent import DQNAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DQN Snake agent demo.")
    parser.add_argument("--model", type=str, default="experiments/checkpoints/dqn_agent.pth")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found at '{model_path}'.")
        sys.exit(1)

    agent = DQNAgent()
    agent.load(str(model_path))
    agent.epsilon = 0.0

    env = SnakeEnv(width=args.width, height=args.height, seed=args.seed)
    renderer = SnakeRenderer(grid_width=args.width, grid_height=args.height)
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
                        env.snake,
                        env.food,
                        env.score,
                        episode=ep,
                        epsilon=0.0,
                        agent_label="DQN Agent",
                        fps=current_fps,
                    )
                    continue

                state = env.get_state()
                action = agent.choose_action(state)
                _, _, done = env.step(action)
                steps += 1

                renderer.render(
                    env.snake,
                    env.food,
                    env.score,
                    episode=ep,
                    epsilon=0.0,
                    agent_label="DQN Agent",
                    fps=current_fps,
                )

            if done:
                for i in range(4):
                    renderer.render_death_flash(
                        renderer.screen,
                        env.snake,
                        env.food,
                        env.score,
                        ep,
                        0.0,
                        "DQN Agent",
                        bright=(i % 2 == 0),
                    )
                    pygame.display.flip()
                    pygame.time.wait(150)

            print(f"Episode {ep}: score={env.score}")
            pygame.time.wait(800)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
