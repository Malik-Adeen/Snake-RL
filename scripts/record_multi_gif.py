import sys, pathlib, random, argparse

import numpy as np
import pygame
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from snake_rl.multi_env import MultiSnakeEnv
from snake_rl.agent import DoubleQAgent
from snake_rl.renderer import MultiSnakeRenderer


def capture_frame(surface: pygame.Surface) -> Image.Image:
    raw = pygame.surfarray.array3d(surface)
    return Image.fromarray(np.transpose(raw, (1, 0, 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record multi-agent Snake gameplay GIF.")
    parser.add_argument("--model_a", type=str, default="experiments/checkpoints/q_table_multi_a.pkl")
    parser.add_argument("--model_b", type=str, default=None)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="assets/demo_multi.gif")
    parser.add_argument("--max_frames", type=int, default=400)
    args = parser.parse_args()

    model_a_path = pathlib.Path(args.model_a)
    if not model_a_path.exists():
        print(f"Error: model file not found at '{model_a_path}'.")
        sys.exit(1)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agent_a = DoubleQAgent(seed=args.seed)
    agent_a.load(model_a_path)
    agent_a.epsilon = 0.0

    if args.model_b is not None:
        model_b_path = pathlib.Path(args.model_b)
        if not model_b_path.exists():
            print(f"Error: model file not found at '{model_b_path}'.")
            sys.exit(1)
        agent_b: DoubleQAgent | None = DoubleQAgent(seed=args.seed + 1)
        agent_b.load(model_b_path)
        agent_b.epsilon = 0.0
        mode_label = "Trained A vs Trained B"
    else:
        agent_b = None
        mode_label = "Trained A vs Random B"

    rng = random.Random(args.seed)
    env = MultiSnakeEnv(width=args.width, height=args.height, seed=args.seed)
    renderer = MultiSnakeRenderer(grid_width=args.width, grid_height=args.height)
    frames: list[Image.Image] = []
    frame_duration_ms = max(50, 1000 // args.fps)

    try:
        for ep in range(1, args.episodes + 1):
            env.reset()
            done = False
            steps = 0
            while not done and steps < args.max_steps:
                action_a = agent_a.choose_action(env.get_state_a())
                action_b = (
                    rng.choice([0, 1, 2]) if agent_b is None else agent_b.choose_action(env.get_state_b())
                )
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
                    epsilon=0.0,
                    mode_label=mode_label,
                    fps=args.fps,
                )
                if len(frames) < args.max_frames:
                    frames.append(capture_frame(renderer.screen))
            winner = (
                "A"
                if env.score_a > env.score_b
                else "B"
                if env.score_b > env.score_a
                else "Draw"
            )
            print(f"Episode {ep}: A={env.score_a} B={env.score_b} -> {winner}")
            pygame.time.wait(800)
    except SystemExit:
        pass
    finally:
        renderer.close()

    if not frames:
        print("No frames captured.")
        return

    width, height = frames[0].size
    resized_size = (max(1, int(width * 0.6)), max(1, int(height * 0.6)))
    resized_frames = [frame.resize(resized_size, Image.LANCZOS) for frame in frames]
    resized_frames[0].save(
        output_path,
        save_all=True,
        append_images=resized_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    size_kb = output_path.stat().st_size // 1024
    print(f"Frames: {len(resized_frames)}")
    print(f"Output: {output_path}")
    print(f"Size: {size_kb} KB")


if __name__ == "__main__":
    main()
