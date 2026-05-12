"""Run a trained tabular Snake agent with pygame visualization."""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import List, Tuple

import pygame

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import SnakeEnv, SnakeRenderer, TabularQAgent
from snake_rl.agent import DoubleQAgent

Position = Tuple[int, int]
DIRECTIONS: List[Position] = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained Snake agent demo.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration rate during demo (0.0 = fully greedy)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Max steps per episode before auto-restart",
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="Load a DoubleQAgent model instead of TabularQAgent",
    )
    parser.add_argument(
        "--show_qvalues",
        action="store_true",
        help="Draw relative-action Q-value overlay around the snake head",
    )
    return parser.parse_args()


def _get_q_values(agent: TabularQAgent | DoubleQAgent, state: Tuple[int, ...]) -> List[float]:
    if isinstance(agent, DoubleQAgent):
        return [
            agent.q_a[state][0] + agent.q_b[state][0],
            agent.q_a[state][1] + agent.q_b[state][1],
            agent.q_a[state][2] + agent.q_b[state][2],
        ]
    return list(agent.q_table[state])


def _draw_q_overlay(
    renderer: SnakeRenderer,
    env: SnakeEnv,
    agent: TabularQAgent | DoubleQAgent,
) -> None:
    if not env.snake:
        return

    state = env.get_state()
    q_values = _get_q_values(agent, state)
    q_min = min(q_values)
    q_max = max(q_values)
    normalized = (
        [(q - q_min) / (q_max - q_min) for q in q_values]
        if q_max != q_min
        else [0.5, 0.5, 0.5]
    )

    ranked = sorted(range(3), key=lambda i: q_values[i])
    raw_colors = {
        ranked[0]: (231, 76, 60),    # red   — worst
        ranked[1]: (243, 156, 18),   # amber — middle
        ranked[2]: (46, 204, 113),   # green — best
    }

    head_x, head_y = env.snake[0]
    cs = renderer.cell_size

    relative_to_abs = {
        0: env.direction_idx,
        1: (env.direction_idx - 1) % 4,
        2: (env.direction_idx + 1) % 4,
    }

    value_font = pygame.font.SysFont("monospace", 16)
    label_font = pygame.font.SysFont("monospace", 11)
    action_names = ["FWD", "LEFT", "RIGHT"]

    for action in (0, 1, 2):
        dir_idx = relative_to_abs[action]
        dx, dy = DIRECTIONS[dir_idx]
        tx = head_x + dx
        ty = head_y + dy

        if tx < 0 or tx >= env.width or ty < 0 or ty >= env.height:
            continue

        color = raw_colors[action]
        norm = normalized[action]

        # Semi-transparent cell highlight
        overlay_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
        alpha = int(80 + norm * 100)   # 80–180 alpha
        overlay_surf.fill((*color, alpha))
        renderer.screen.blit(overlay_surf, (tx * cs, ty * cs))

        # Colored border around target cell
        pygame.draw.rect(
            renderer.screen, color,
            pygame.Rect(tx * cs + 2, ty * cs + 2, cs - 4, cs - 4),
            width=3,
            border_radius=6,
        )

        # Arrow from head center to target cell center
        head_cx = head_x * cs + cs // 2
        head_cy = head_y * cs + cs // 2
        tgt_cx  = tx * cs + cs // 2
        tgt_cy  = ty * cs + cs // 2
        pygame.draw.line(renderer.screen, color,
                         (head_cx, head_cy), (tgt_cx, tgt_cy), 3)
        # Arrowhead triangle
        perp_x, perp_y = -dy, dx
        tip = (tgt_cx, tgt_cy)
        base_cx = head_cx + (tgt_cx - head_cx) * 3 // 4
        base_cy = head_cy + (tgt_cy - head_cy) * 3 // 4
        pts = [
            tip,
            (base_cx + perp_x * 7, base_cy + perp_y * 7),
            (base_cx - perp_x * 7, base_cy - perp_y * 7),
        ]
        pygame.draw.polygon(renderer.screen, color, pts)

        # Q-value label with dark pill background
        val_str  = f"{norm:.2f}"
        name_str = action_names[action]
        val_surf  = value_font.render(val_str,  True, (255, 255, 255))
        name_surf = label_font.render(name_str, True, color)
        lx = tx * cs + cs // 2 - val_surf.get_width() // 2
        ly = ty * cs + cs // 2 - val_surf.get_height() // 2 - 2
        bg_rect = pygame.Rect(lx - 4, ly - 2,
                              val_surf.get_width() + 8,
                              val_surf.get_height() + name_surf.get_height() + 6)
        bg_surf = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surf.fill((10, 10, 10, 190))
        renderer.screen.blit(bg_surf, (bg_rect.x, bg_rect.y))
        renderer.screen.blit(val_surf,  (lx, ly))
        renderer.screen.blit(name_surf, (lx, ly + val_surf.get_height() + 1))

    # HUD badge
    badge_font = pygame.font.SysFont("monospace", 13)
    hud_y = renderer.grid_height * renderer.cell_size
    badge = badge_font.render("Q-values: ON", True, (120, 230, 150))
    renderer.screen.blit(badge, (12, hud_y + 60))


def main() -> None:
    args = parse_args()
    if args.model is None:
        args.model = (
            "experiments/checkpoints/q_table_double.pkl"
            if args.double
            else "experiments/checkpoints/q_table.pkl"
        )
    rng = random.Random(args.seed)
    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found at '{model_path}'. Train first or pass --model.")
        sys.exit(1)

    agent = DoubleQAgent(seed=args.seed) if args.double else TabularQAgent(seed=args.seed)
    agent.load(model_path)
    agent.epsilon = 0.0  # always greedy; demo handles its own epsilon via rng

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
                    if args.show_qvalues:
                        renderer.render_to_surface(
                            renderer.screen,
                            env.snake, env.food, env.score,
                            ep, args.epsilon, "Trained Agent",
                        )
                        _draw_q_overlay(renderer, env, agent)
                        pygame.display.flip()
                        renderer.clock.tick(current_fps)
                    else:
                        renderer.render(
                            env.snake, env.food, env.score,
                            episode=ep, epsilon=args.epsilon,
                            agent_label="Trained Agent", fps=current_fps,
                        )
                    continue

                state = env.get_state()
                if rng.random() < args.epsilon:
                    action = rng.choice([0, 1, 2])
                else:
                    action = agent.choose_action(state)

                _, _, done = env.step(action)
                steps += 1

                if args.show_qvalues:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise SystemExit
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                            raise SystemExit
                    renderer.render_to_surface(
                        renderer.screen,
                        env.snake, env.food, env.score,
                        ep, args.epsilon, "Trained Agent",
                    )
                    _draw_q_overlay(renderer, env, agent)
                    pygame.display.flip()
                    renderer.clock.tick(current_fps)
                else:
                    renderer.render(
                        env.snake, env.food, env.score,
                        episode=ep, epsilon=args.epsilon,
                        agent_label="Trained Agent", fps=current_fps,
                    )

            # Death animation — 4 alternating flash frames
            if done:
                for i in range(4):
                    bright = (i % 2 == 0)
                    renderer.render_death_flash(
                        renderer.screen,
                        env.snake,
                        env.food,
                        env.score,
                        ep,
                        args.epsilon,
                        "Trained Agent",
                        bright=bright,
                    )
                    pygame.display.flip()
                    pygame.time.wait(150)

            if steps >= args.max_steps:
                print(f"Episode {ep}: score={env.score} (timeout)")
            else:
                print(f"Episode {ep}: score={env.score}")
            pygame.time.wait(1000)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
