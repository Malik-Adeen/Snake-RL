import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main() -> None:
    run_labels = [
        "Run 1\nBaseline\n7-bit",
        "Run 2\n+Shaping\n7-bit",
        "Run 3\n8-bit\n2k eps",
        "Run 4\n8-bit\n5k eps",
        "Run 5\nDouble Q\n8-bit",
        "Run 6\nDouble Q\n12-bit",
    ]
    train_means = [1.74, 2.16, 1.58, 1.81, 2.34, 6.50]
    eval_means = [None, None, None, None, None, 11.21]
    max_scores = [11, 15, 15, 15, 14, 24]

    episodes_checkpoints = [0, 500, 1000, 3000, 5000]
    trained_vs_random_wr = [50.0, 47.4, 83.6, 93.0, 95.0]
    selfplay_wr = [50.0, 23.4, 35.8, 44.2, 44.2]

    plots_dir = os.path.join("experiments", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    progression_path = os.path.join(plots_dir, "report_progression.png")
    winrate_detail_path = os.path.join(plots_dir, "report_winrate_detail.png")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    x_runs = np.arange(len(run_labels))
    bars = ax_left.bar(
        x_runs,
        train_means,
        color="steelblue",
        label="Training mean (last 100)",
    )
    ax_left.set_xticks(x_runs)
    ax_left.set_xticklabels(run_labels, fontsize=8)
    ax_left.set_ylabel("Mean Score (last 100 eps)")
    ax_left.set_title("Single-Agent Score Progression (Runs 1–6)")
    ax_left.grid(alpha=0.3)

    ax_left_right = ax_left.twinx()
    max_line, = ax_left_right.plot(
        x_runs,
        max_scores,
        color="tomato",
        marker="o",
        linestyle="--",
        label="Max score",
    )
    ax_left_right.set_ylabel("Max Score")

    for idx, eval_value in enumerate(eval_means):
        if eval_value is not None:
            ax_left.scatter(idx, eval_value, color="gold", marker="*", s=200, zorder=5)
            ax_left.annotate(
                f"Eval: {eval_value:.2f}",
                xy=(idx, eval_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    ax_left.legend(
        [bars, max_line],
        ["Training mean (last 100)", "Max score"],
        loc="upper left",
    )

    ax_right.plot(
        episodes_checkpoints,
        trained_vs_random_wr,
        color="blue",
        marker="o",
        linestyle="-",
        label="Trained vs Random (Agent A)",
    )
    ax_right.plot(
        episodes_checkpoints,
        selfplay_wr,
        color="orange",
        marker="s",
        linestyle="-",
        label="Self-Play (Agent A)",
    )
    ax_right.axhline(50, color="grey", linestyle="--", alpha=0.5, label="50% baseline")
    ax_right.set_xlabel("Training Episode")
    ax_right.set_ylabel("Win Rate (%)")
    ax_right.set_ylim(0, 100)
    ax_right.set_title("Multi-Agent Win Rate Convergence")
    ax_right.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax_right.legend(loc="lower right")
    ax_right.grid(alpha=0.3)

    plt.suptitle("Snake RL — Experiment Progression", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(progression_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        episodes_checkpoints,
        trained_vs_random_wr,
        color="blue",
        marker="o",
        linestyle="-",
        label="Trained vs Random (Agent A)",
    )
    ax.plot(
        episodes_checkpoints,
        selfplay_wr,
        color="orange",
        marker="s",
        linestyle="-",
        label="Self-Play (Agent A)",
    )
    ax.axhline(50, color="grey", linestyle="--", alpha=0.5, label="50% baseline")
    ax.annotate(
        "95.0%",
        xy=(5000, 95.0),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
    ax.annotate(
        "44.2%",
        xy=(5000, 44.2),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
    )
    ax.set_title("Multi-Agent Win Rate — Trained vs Random vs Self-Play")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(winrate_detail_path, dpi=150)
    plt.close(fig)

    print(progression_path)
    print(winrate_detail_path)


if __name__ == "__main__":
    main()
