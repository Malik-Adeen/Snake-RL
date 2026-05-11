"""Entry point for multi-agent training in competitive Snake."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, cast

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import multi_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two agents in MultiSnakeEnv.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="trained_vs_random",
        choices=["trained_vs_random", "self_play"],
    )
    parser.add_argument(
        "--save_path_a",
        type=str,
        default="experiments/checkpoints/q_table_multi_a.pkl",
    )
    parser.add_argument(
        "--save_path_b",
        type=str,
        default="experiments/checkpoints/q_table_multi_b.pkl",
    )
    parser.add_argument("--plot_dir", type=str, default="experiments/plots")
    return parser.parse_args()


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return np.array([], dtype=float)
    weights = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, weights, mode="valid")


def main() -> None:
    args = parse_args()
    save_path_a = pathlib.Path(args.save_path_a)
    save_path_b = pathlib.Path(args.save_path_b)
    plot_dir = pathlib.Path(args.plot_dir)

    save_path_a.parent.mkdir(parents=True, exist_ok=True)
    save_path_b.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = multi_train(
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        mode=args.mode,
        save_path_a=str(save_path_a),
        save_path_b=str(save_path_b),
    )

    scores_a = cast(List[int], results["scores_a"])
    scores_b = cast(List[int], results["scores_b"])
    win_rate_a = float(results["win_rate_a"])

    last_100_a = scores_a[-100:]
    last_100_b = scores_b[-100:]
    mean_last_100_a = float(np.mean(last_100_a)) if last_100_a else 0.0
    mean_last_100_b = float(np.mean(last_100_b)) if last_100_b else 0.0
    max_score_a = max(scores_a) if scores_a else 0
    max_score_b = max(scores_b) if scores_b else 0

    print(f"Win rate Agent A: {win_rate_a:.2f}%")
    print(f"Mean score last 100 | A: {mean_last_100_a:.2f} | B: {mean_last_100_b:.2f}")
    print(f"Max score           | A: {max_score_a} | B: {max_score_b}")

    plt.style.use("seaborn-v0_8-darkgrid")

    scores_a_arr = np.array(scores_a, dtype=float)
    scores_b_arr = np.array(scores_b, dtype=float)
    episodes_x = np.arange(1, len(scores_a_arr) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, scores_a_arr, color="lightgreen", alpha=0.4, label="Agent A")
    plt.plot(episodes_x, scores_b_arr, color="lightcoral", alpha=0.4, label="Agent B")
    ma_window = 100
    ma_a = _moving_average(scores_a_arr, window=ma_window)
    ma_b = _moving_average(scores_b_arr, window=ma_window)
    if ma_a.size > 0:
        ma_x = np.arange(ma_window, len(scores_a_arr) + 1)
        plt.plot(ma_x, ma_a, color="darkgreen", linewidth=2, label="Agent A MA(100)")
    if ma_b.size > 0:
        ma_x = np.arange(ma_window, len(scores_b_arr) + 1)
        plt.plot(ma_x, ma_b, color="darkred", linewidth=2, label="Agent B MA(100)")
    plt.title(f"Multi-Agent Scores ({args.mode})")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "multi_scores.png", dpi=150)
    plt.close()

    wins_a_arr = np.array(
        [1.0 if a_score > b_score else 0.0 for a_score, b_score in zip(scores_a, scores_b)],
        dtype=float,
    )
    win_window = 200
    rolling_win = _moving_average(wins_a_arr, window=win_window) * 100.0
    rolling_x = np.arange(win_window, len(wins_a_arr) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_x, rolling_win, color="tab:blue", linewidth=2)
    plt.axhline(50.0, color="black", linestyle="--", linewidth=1.5)
    plt.title("Agent A Win Rate (rolling 200)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(plot_dir / "multi_winrate.png", dpi=150)
    plt.close()

    print(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
