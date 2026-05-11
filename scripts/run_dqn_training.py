import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib.pyplot as plt
import numpy as np

from snake_rl.dqn_agent import DQNAgent
from snake_rl.dqn_train import dqn_train
from snake_rl.env import SnakeEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate DQN on Snake.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="experiments/checkpoints/dqn_agent.pth")
    parser.add_argument("--eval_seed", type=int, default=99)
    parser.add_argument("--eval_eps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon_decay", type=float, default=0.997)
    args = parser.parse_args()

    metrics: dict[str, list] = dqn_train(
        episodes=args.episodes,
        seed=args.seed,
        save_path=args.save_path,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
    )

    agent = DQNAgent()
    agent.load(args.save_path)
    agent.epsilon = 0.0

    eval_env = SnakeEnv(seed=args.eval_seed)
    eval_scores: list[int] = []
    for _ in range(args.eval_eps):
        eval_env.reset()
        state = eval_env.get_state()
        done = False
        steps = 0
        while not done and steps < 1000:
            action = agent.choose_action(state)
            _, _, done = eval_env.step(action)
            state = eval_env.get_state()
            steps += 1
        eval_scores.append(eval_env.score)

    dqn_mean = float(np.mean(eval_scores))
    dqn_std = float(np.std(eval_scores))
    dqn_max = int(max(eval_scores))
    tab_mean = 11.21
    tab_std = 4.92
    tab_max = 24.0

    print(f"=== DQN Evaluation (seed={args.eval_seed}) ===")
    print(f"Mean score: {dqn_mean:.2f}")
    print(f"Std score:  {dqn_std:.2f}")
    print(f"Max score:  {dqn_max}")

    plots_dir = os.path.join("experiments", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_scores = np.array(metrics["scores"], dtype=float)
    avg_scores = np.array(metrics["avg_scores"], dtype=float)
    losses = np.array(metrics["losses"], dtype=float)
    epsilons = np.array(metrics["epsilons"], dtype=float)

    plot1_path = os.path.join(plots_dir, "dqn_learning_curve.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x_scores = np.arange(1, len(all_scores) + 1)
    raw_line, = axes[0].plot(
        x_scores, all_scores, color="lightblue", alpha=0.4, linewidth=1.0, label="Raw score"
    )
    if len(avg_scores) > 0:
        x_avg = np.arange(100, 100 * len(avg_scores) + 1, 100)
        avg_line, = axes[0].plot(x_avg, avg_scores, color="darkblue", linewidth=2.0, label="Mean (100 ep)")
    else:
        avg_line, = axes[0].plot([], [], color="darkblue", linewidth=2.0, label="Mean (100 ep)")
    axes[0].set_title("DQN Learning Curve")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")
    axes[0].legend([raw_line, avg_line], ["Raw score", "Mean (100 ep)"])

    x_losses = np.arange(1, len(losses) + 1)
    axes[1].plot(x_losses, losses, color="orange", alpha=0.6, linewidth=1.0)
    axes[1].set_title("DQN Training Loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("MSE Loss")

    plt.tight_layout()
    fig.savefig(plot1_path)
    plt.close(fig)

    plot2_path = os.path.join(plots_dir, "dqn_epsilon.png")
    fig, ax = plt.subplots()
    x_eps = np.arange(1, len(epsilons) + 1)
    ax.plot(x_eps, epsilons, color="green")
    ax.set_title("DQN Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    fig.tight_layout()
    fig.savefig(plot2_path)
    plt.close(fig)

    plot3_path = os.path.join(plots_dir, "dqn_vs_tabular.png")
    categories = ["Mean Score", "Max Score"]
    x = np.arange(len(categories))
    width = 0.35

    dqn_values = [dqn_mean, float(dqn_max)]
    tab_values = [tab_mean, tab_max]
    dqn_yerr = [dqn_std, 0.0]
    tab_yerr = [tab_std, 0.0]

    fig, ax = plt.subplots()
    bars_dqn = ax.bar(
        x - width / 2,
        dqn_values,
        width,
        color="steelblue",
        label="DQN",
        yerr=dqn_yerr,
        capsize=5,
    )
    bars_tab = ax.bar(
        x + width / 2,
        tab_values,
        width,
        color="coral",
        label="Double Q Tabular",
        yerr=tab_yerr,
        capsize=5,
    )

    ax.set_title("DQN vs Double Q-Learning")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    for bar in list(bars_dqn) + list(bars_tab):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(plot3_path)
    plt.close(fig)

    print(f"Saved plot: {plot1_path}")
    print(f"Saved plot: {plot2_path}")
    print(f"Saved plot: {plot3_path}")


if __name__ == "__main__":
    main()
