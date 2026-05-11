"""Record a gameplay GIF from the trained Snake agent.

Captures pygame frames during a demo run and saves an animated GIF
to assets/ using Pillow. Run this to generate footage for the README.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np
import pygame
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import SnakeEnv, SnakeRenderer
from snake_rl.agent import DoubleQAgent
from scripts.run_demo import _draw_q_overlay, _get_q_values   # reuse overlay logic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a gameplay GIF.")
    parser.add_argument("--model", type=str,
                        default="experiments/checkpoints/q_table_double.pkl")
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument("--width",  type=int,   default=10)
    parser.add_argument("--height", type=int,   default=10)
    parser.add_argument("--fps",    type=int,   default=8)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--epsilon",   type=float, default=0.05)
    parser.add_argument("--seed",      type=int,   default=3)
    parser.add_argument("--show_qvalues", action="store_true", default=True)
    parser.add_argument("--output", type=str,
                        default="assets/demo_qoverlay.gif")
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Cap total frames to keep GIF size reasonable")
    return parser.parse_args()


def capture_frame(screen: pygame.Surface) -> Image.Image:
    """Capture current pygame surface as a PIL Image."""
    raw = pygame.surfarray.array3d(screen)
    # surfarray gives (width, height, 3) — PIL wants (height, width, 3)
    img = Image.fromarray(np.transpose(raw, (1, 0, 2)))
    return img


def main() -> None:
    args = parse_args()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found at '{model_path}'.")
        sys.exit(1)

    agent = DoubleQAgent(seed=args.seed)
    agent.load(model_path)
    agent.epsilon = 0.0

    env = SnakeEnv(width=args.width, height=args.height, seed=args.seed)
    renderer = SnakeRenderer(grid_width=args.width, grid_height=args.height)
    rng = random.Random(args.seed)

    frames: list[Image.Image] = []
    frame_duration_ms = max(50, 1000 // args.fps)

    print(f"Recording {args.episodes} episodes at {args.fps} fps...")
    print(f"Output: {output_path}")

    try:
        for ep in range(1, args.episodes + 1):
            env.reset()
            done = False
            steps = 0

            while not done and steps < args.max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        raise SystemExit

                state = env.get_state()
                action = (
                    rng.choice([0, 1, 2])
                    if rng.random() < args.epsilon
                    else agent.choose_action(state)
                )
                _, _, done = env.step(action)
                steps += 1

                renderer.render_to_surface(
                    renderer.screen,
                    env.snake, env.food, env.score,
                    ep, args.epsilon, "Trained Agent",
                )

                if args.show_qvalues:
                    _draw_q_overlay(renderer, env, agent)

                pygame.display.flip()
                renderer.clock.tick(args.fps)

                if len(frames) < args.max_frames:
                    frames.append(capture_frame(renderer.screen))

            print(f"  Episode {ep}: score={env.score} | frames so far: {len(frames)}")
            pygame.time.wait(800)

    finally:
        renderer.close()

    if not frames:
        print("No frames captured.")
        return

    # Resize to 50% for smaller GIF file
    w, h = frames[0].size
    small_frames = [f.resize((w // 2, h // 2), Image.LANCZOS) for f in frames]

    print(f"Saving {len(small_frames)} frames → {output_path} ...")
    small_frames[0].save(
        output_path,
        save_all=True,
        append_images=small_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = output_path.stat().st_size // 1024
    print(f"Done. GIF saved: {size_kb} KB")


if __name__ == "__main__":
    main()
