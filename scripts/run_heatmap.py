"""Policy heatmap visualizer for trained Snake agents."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import SnakeEnv, TabularQAgent
from snake_rl.agent import DoubleQAgent

Position = Tuple[int, int]
State = Tuple[int, int, int, int, int, int, int, int, int, int, int, int]

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
ACTION_COLORS = {
    0: "#2ecc71",
    1: "#3498db",
    2: "#e74c3c",
}


def _is_wall(x: int, y: int, width: int, height: int) -> bool:
    return x < 0 or x >= width or y < 0 or y >= height


def _is_collision(pos: Position, head: Position, width: int, height: int) -> bool:
    return _is_wall(pos[0], pos[1], width, height) or pos == head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate policy heatmaps for trained Snake agents.")
    parser.add_argument("--model", type=str, default="experiments/checkpoints/q_table_double.pkl")
    parser.add_argument("--double", action="store_true", help="Use DoubleQAgent instead of TabularQAgent")
    parser.add_argument("--food_x", type=int, default=5)
    parser.add_argument("--food_y", type=int, default=5)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--output", type=str, default="experiments/plots/policy_heatmap.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = SnakeEnv  # imported as requested

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.double:
        agent = DoubleQAgent(seed=0)
    else:
        agent = TabularQAgent(seed=0)

    try:
        agent.load(args.model)
    except FileNotFoundError:
        print(f"Error: model file not found at '{args.model}'.")
        sys.exit(1)

    agent.epsilon = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    labels = ["Facing Right", "Facing Down", "Facing Left", "Facing Up"]

    for dir_idx, ax in enumerate(axes.flatten()):
        for y in range(args.height):
            for x in range(args.width):
                if x == args.food_x and y == args.food_y:
                    continue

                head = (x, y)
                neck = (x - 1, y)
                tail = (0, 0)

                straight_dir = DIRECTIONS[dir_idx]
                left_dir = DIRECTIONS[(dir_idx - 1) % 4]
                right_dir = DIRECTIONS[(dir_idx + 1) % 4]

                straight_pos = (x + straight_dir[0], y + straight_dir[1])
                left_pos = (x + left_dir[0], y + left_dir[1])
                right_pos = (x + right_dir[0], y + right_dir[1])

                danger_straight = int(
                    _is_collision(straight_pos, neck, args.width, args.height)
                    or _is_collision(straight_pos, tail, args.width, args.height)
                )
                danger_left = int(
                    _is_collision(left_pos, neck, args.width, args.height)
                    or _is_collision(left_pos, tail, args.width, args.height)
                )
                danger_right = int(
                    _is_collision(right_pos, neck, args.width, args.height)
                    or _is_collision(right_pos, tail, args.width, args.height)
                )

                food_left = int(args.food_x < x)
                food_right = int(args.food_x > x)
                food_up = int(args.food_y < y)
                food_down = int(args.food_y > y)
                food_close = int(abs(args.food_x - x) + abs(args.food_y - y) <= 3)

                tail_left = int(0 < x)
                tail_right = int(0 > x)
                tail_up = int(0 < y)
                tail_down = int(0 > y)

                state: State = (
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

                action = agent.choose_action(state)

                rect = mpatches.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=mcolors.to_rgba(ACTION_COLORS[action], alpha=0.7),
                    edgecolor="none",
                )
                ax.add_patch(rect)

                move_dir = (
                    straight_dir if action == 0 else left_dir if action == 1 else right_dir
                )
                cx, cy = x + 0.5, y + 0.5
                tx, ty = cx + 0.25 * move_dir[0], cy + 0.25 * move_dir[1]
                ax.annotate(
                    "",
                    xy=(tx, ty),
                    xytext=(cx, cy),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
                )

        food_marker = plt.Circle((args.food_x + 0.5, args.food_y + 0.5), 0.25, color="red")
        ax.add_patch(food_marker)

        ax.set_xlim(0, args.width)
        ax.set_ylim(0, args.height)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(0, args.width + 1, 1))
        ax.set_yticks(np.arange(0, args.height + 1, 1))
        ax.grid(True, color="gray", linewidth=0.5)
        ax.set_title(labels[dir_idx])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    legend_handles = [
        mpatches.Patch(color=ACTION_COLORS[0], label="Straight"),
        mpatches.Patch(color=ACTION_COLORS[1], label="Turn Left"),
        mpatches.Patch(color=ACTION_COLORS[2], label="Turn Right"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {args.output}")


if __name__ == "__main__":
    main()
