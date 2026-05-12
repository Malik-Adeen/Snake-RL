"""Record random vs trained Snake side-by-side and save as an animated GIF."""

import sys, os, pathlib, random, argparse

import numpy as np
import pygame
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from snake_rl import SnakeEnv, SnakeRenderer
from snake_rl.agent import DoubleQAgent


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


def capture_frame(screen: pygame.Surface) -> Image.Image:
    raw = pygame.surfarray.array3d(screen)
    return Image.fromarray(np.transpose(raw, (1, 0, 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record side-by-side Snake gameplay GIF.")
    parser.add_argument("--model", type=str, default="experiments/checkpoints/q_table_double.pkl")
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="assets/demo_sidebyside.gif")
    parser.add_argument("--max_frames", type=int, default=400)
    args = parser.parse_args()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found at '{model_path}'.")
        sys.exit(1)

    agent = DoubleQAgent(seed=args.seed)
    agent.load(model_path)
    agent.epsilon = 0.0

    cell_size = 40
    panel_w = args.width * cell_size
    window_w = 2 * panel_w + 12
    window_h = args.height * cell_size + 80

    pygame.init()
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("Snake RL — Side by Side GIF Recorder")
    left_surface = screen.subsurface((0, 0, panel_w, window_h))
    right_surface = screen.subsurface((panel_w + 12, 0, panel_w, window_h))
    left_renderer = _build_panel_renderer(left_surface, args.width, args.height, cell_size)
    right_renderer = _build_panel_renderer(right_surface, args.width, args.height, cell_size)
    clock = pygame.time.Clock()

    rng = random.Random(args.seed)
    frames: list[Image.Image] = []
    frame_duration_ms = max(50, 1000 // args.fps)
    stop_requested = False

    try:
        for ep in range(1, args.episodes + 1):
            episode_seed = args.seed + ep
            env_random = SnakeEnv(width=args.width, height=args.height, seed=episode_seed)
            env_trained = SnakeEnv(width=args.width, height=args.height, seed=episode_seed)
            env_random.reset()
            env_trained.reset()

            done_random = False
            done_trained = False
            steps = 0

            while (not done_random or not done_trained) and steps < args.max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop_requested = True
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        stop_requested = True
                        break
                if stop_requested:
                    break

                if not done_random:
                    action_random = rng.choice([0, 1, 2])
                    _, _, done_random = env_random.step(action_random)

                if not done_trained:
                    action_trained = agent.choose_action(env_trained.get_state())
                    _, _, done_trained = env_trained.step(action_trained)

                steps += 1

                left_renderer.render_to_surface(
                    left_surface,
                    env_random.snake,
                    env_random.food,
                    env_random.score,
                    ep,
                    agent.epsilon,
                    "Random",
                )
                right_renderer.render_to_surface(
                    right_surface,
                    env_trained.snake,
                    env_trained.food,
                    env_trained.score,
                    ep,
                    agent.epsilon,
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
                    epsilon=agent.epsilon,
                )
                pygame.display.flip()
                clock.tick(args.fps)

                if len(frames) < args.max_frames:
                    frames.append(capture_frame(screen))

            print(f"Episode {ep}: random={env_random.score} trained={env_trained.score}")
            if stop_requested:
                break
            pygame.time.wait(600)
    finally:
        pygame.quit()

    if not frames:
        print("No frames captured.")
        return

    width, height = frames[0].size
    resized = (max(1, int(width * 0.6)), max(1, int(height * 0.6)))
    small_frames = [frame.resize(resized, Image.LANCZOS) for frame in frames]

    small_frames[0].save(
        output_path,
        save_all=True,
        append_images=small_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = output_path.stat().st_size // 1024
    print(f"Frames: {len(small_frames)}")
    print(f"Output: {output_path}")
    print(f"Size: {size_kb} KB")


if __name__ == "__main__":
    main()
