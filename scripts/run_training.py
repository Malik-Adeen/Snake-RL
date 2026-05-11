"""Main training entry point for Snake tabular Q-learning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl.agent import DoubleQAgent
from snake_rl.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning Snake agent.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon_decay", type=float, default=0.997)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments/checkpoints/q_table.pkl",
    )
    parser.add_argument("--plot_dir", type=str, default="experiments/plots")
    parser.add_argument(
        "--double",
        action="store_true",
        help="Use Double Q-Learning instead of standard Q-Learning",
    )
    return parser.parse_args()


def _moving_average(values: List[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array([], dtype=float)
    weights = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.array(values, dtype=float), weights, mode="valid")


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path)
    plot_dir = Path(args.plot_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    epsilon_min = 0.05
    results = train(
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        save_path=str(save_path),
        use_double=args.double,
    )

    scores = cast(List[int], results["scores"])
    epsilons = cast(List[float], results["epsilons"])
    q_table_sizes = cast(List[int], results["q_table_sizes"])

    max_score = max(scores) if scores else 0
    mean_last_100 = float(np.mean(scores[-100:])) if scores else 0.0
    mean_last_500 = float(np.mean(scores[-500:])) if scores else 0.0
    states_visited = q_table_sizes[-1] if q_table_sizes else 0

    print(f"Max score achieved: {max_score}")
    print(f"Mean score (last 100): {mean_last_100:.2f}")
    print(f"Mean score (last 500): {mean_last_500:.2f}")
    print(f"Total unique states visited: {states_visited}")

    plt.style.use("seaborn-v0_8-darkgrid")
    episodes_x = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, scores, color="lightgray", alpha=0.4, label="Score")
    ma100 = _moving_average([float(v) for v in scores], window=100)
    if ma100.size > 0:
        ma_x = np.arange(100, len(scores) + 1)
        plt.plot(ma_x, ma100, color="tab:blue", linewidth=2, label="Moving Avg (100)")
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.text(
        0.02,
        0.95,
        f"Mean last 100: {mean_last_100:.2f}",
        transform=plt.gca().transAxes,
        va="top",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "learning_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(epsilons) + 1), epsilons, color="tab:orange")
    plt.axhline(y=epsilon_min, color="tab:red", linestyle="--", linewidth=1.5)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.tight_layout()
    plt.savefig(plot_dir / "epsilon_decay.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(q_table_sizes) + 1), q_table_sizes, color="tab:green")
    plt.title("Q-Table Growth (Visited States)")
    plt.xlabel("Episode")
    plt.ylabel("Unique States")
    plt.tight_layout()
    plt.savefig(plot_dir / "qtable_growth.png", dpi=150)
    plt.close()

    print(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
