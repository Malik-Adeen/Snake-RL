"""Run Snake training across multiple seeds and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snake_rl import train
from snake_rl.agent import DoubleQAgent, TabularQAgent

AgentType = DoubleQAgent | TabularQAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiseed Snake training evaluation.")
    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--episodes", type=int, default=15000)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--out", type=str, default="experiments/multiseed_results.json")
    return parser.parse_args()


def _parse_seeds(seeds_arg: str) -> list[int]:
    seeds = [token.strip() for token in seeds_arg.split(",") if token.strip()]
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return [int(seed) for seed in seeds]


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def _fmt_int(value: float) -> str:
    return f"{int(round(value))}"


def main() -> None:
    args = parse_args()
    _ = AgentType
    seeds = _parse_seeds(args.seeds)

    save_dir = pathlib.Path(args.save_dir)
    out_path = pathlib.Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "mean_last_100",
        "mean_last_500",
        "max_score",
        "final_q_table_size",
        "final_epsilon",
    ]

    per_seed: dict[str, dict[str, float]] = {}
    metric_values: dict[str, list[float]] = {name: [] for name in metric_names}

    for seed in seeds:
        save_path = save_dir / f"q_table_seed{seed}.pkl"
        results = train(
            episodes=args.episodes,
            epsilon_decay=args.epsilon_decay,
            seed=seed,
            save_path=str(save_path),
            use_double=args.double,
        )

        scores = [int(v) for v in results["scores"]]
        q_table_sizes = [int(v) for v in results["q_table_sizes"]]
        epsilons = [float(v) for v in results["epsilons"]]

        metrics = {
            "mean_last_100": float(np.mean(scores[-100:])) if scores else 0.0,
            "mean_last_500": float(np.mean(scores[-500:])) if scores else 0.0,
            "max_score": float(max(scores) if scores else 0),
            "final_q_table_size": float(q_table_sizes[-1] if q_table_sizes else 0),
            "final_epsilon": float(epsilons[-1] if epsilons else 0.0),
        }

        per_seed[str(seed)] = metrics
        for metric_name in metric_names:
            metric_values[metric_name].append(metrics[metric_name])

    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.array(metric_values[metric_name], dtype=float)
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    display_rows = [
        ("Mean(last 100)", "mean_last_100", _fmt_float),
        ("Mean(last 500)", "mean_last_500", _fmt_float),
        ("Max Score", "max_score", _fmt_int),
        ("States Visited", "final_q_table_size", _fmt_int),
        ("Final Epsilon", "final_epsilon", _fmt_float),
    ]

    metric_col_width = 16
    value_col_width = 12
    header = ["Metric"] + [f"Seed {seed}" for seed in seeds] + ["Mean ± Std"]
    header_line = " | ".join(
        [f"{header[0]:<{metric_col_width}}"]
        + [f"{name:>{value_col_width}}" for name in header[1:]]
    )
    print(header_line)
    print("-" * len(header_line))

    for row_name, metric_key, fmt_fn in display_rows:
        seed_cols = [fmt_fn(per_seed[str(seed)][metric_key]) for seed in seeds]
        mean_val = summary[metric_key]["mean"]
        std_val = summary[metric_key]["std"]
        summary_text = f"{fmt_fn(mean_val)} ± {fmt_fn(std_val)}"
        row_text = " | ".join(
            [f"{row_name:<{metric_col_width}}"]
            + [f"{cell:>{value_col_width}}" for cell in seed_cols]
            + [f"{summary_text:>{value_col_width}}"]
        )
        print(row_text)

    output = {
        "config": {
            "episodes": args.episodes,
            "epsilon_decay": args.epsilon_decay,
            "double": args.double,
            "seeds": seeds,
        },
        "per_seed": per_seed,
        "summary": summary,
    }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
