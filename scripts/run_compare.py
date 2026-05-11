"""Compare trained and random Snake agents with episode-level statistics."""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import SnakeEnv, TabularQAgent
from snake_rl.agent import DoubleQAgent

State = Tuple[int, ...]
Policy = Callable[[State], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare random vs trained Snake policies.")
    parser.add_argument("--model", type=str, default="experiments/checkpoints/q_table.pkl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--double",
        action="store_true",
        help="Load a DoubleQAgent model instead of TabularQAgent",
    )
    return parser.parse_args()


def run_policy(
    policy: Policy,
    episodes: int,
    width: int,
    height: int,
    seed_base: int,
    max_steps: int = 2000,
) -> Dict[str, List[int] | int]:
    scores: List[int] = []
    episode_lengths: List[int] = []
    max_score: int = 0

    for ep in range(episodes):
        env = SnakeEnv(width=width, height=height, seed=seed_base + ep)
        env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            state = env.get_state()
            action = policy(state)
            _, _, done = env.step(action)
            steps += 1

        scores.append(env.score)
        episode_lengths.append(steps)
        if env.score > max_score:
            max_score = env.score

    return {
        "scores": scores,
        "episode_lengths": episode_lengths,
        "max_score": max_score,
    }


def _pct_ge(scores: np.ndarray, threshold: int) -> float:
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= threshold) * 100.0)


def _format_score(value: float) -> str:
    return f"{value:>12.2f}"


def _format_count(value: int) -> str:
    return f"{value:>12d}"


def _format_percent(value: float) -> str:
    return f"{value:>11.1f}%"


def main() -> None:
    args = parse_args()

    agent = DoubleQAgent(seed=args.seed) if args.double else TabularQAgent(seed=args.seed)
    try:
        agent.load(args.model)
    except FileNotFoundError:
        print(f"Error: model file not found at '{args.model}'. Train first or pass --model.")
        sys.exit(1)
    agent.epsilon = 0.0

    random.seed(args.seed)

    def random_policy(_: State) -> int:
        return random.choice([0, 1, 2])

    def trained_policy(state: State) -> int:
        return agent.choose_action(state)

    random_results = run_policy(
        random_policy,
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        seed_base=args.seed,
    )
    trained_results = run_policy(
        trained_policy,
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        seed_base=args.seed,
    )

    random_scores = np.array(random_results["scores"], dtype=float)
    trained_scores = np.array(trained_results["scores"], dtype=float)
    random_lengths = np.array(random_results["episode_lengths"], dtype=float)
    trained_lengths = np.array(trained_results["episode_lengths"], dtype=float)

    print("Metric            | Random Agent | Trained Agent")
    print("Mean Score        |"
          f"{_format_score(float(np.mean(random_scores)))} |"
          f"{_format_score(float(np.mean(trained_scores)))}")
    print("Std Score         |"
          f"{_format_score(float(np.std(random_scores)))} |"
          f"{_format_score(float(np.std(trained_scores)))}")
    print("Max Score         |"
          f"{_format_count(int(random_results['max_score']))} |"
          f"{_format_count(int(trained_results['max_score']))}")
    print("Mean Ep Length    |"
          f"{_format_score(float(np.mean(random_lengths)))} |"
          f"{_format_score(float(np.mean(trained_lengths)))}")
    print("Score > 0 (%)     |"
          f"{_format_percent(_pct_ge(random_scores, 1))} |"
          f"{_format_percent(_pct_ge(trained_scores, 1))}")
    print("Score >= 5 (%)    |"
          f"{_format_percent(_pct_ge(random_scores, 5))} |"
          f"{_format_percent(_pct_ge(trained_scores, 5))}")
    print("Score >= 10 (%)   |"
          f"{_format_percent(_pct_ge(random_scores, 10))} |"
          f"{_format_percent(_pct_ge(trained_scores, 10))}")


if __name__ == "__main__":
    main()
