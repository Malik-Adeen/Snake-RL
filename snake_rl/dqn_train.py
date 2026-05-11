from __future__ import annotations

import os

import numpy as np

from snake_rl.dqn_agent import DQNAgent
from snake_rl.env import SnakeEnv


def dqn_train(
    episodes: int = 3000,
    seed: int = 42,
    max_steps: int = 1000,
    save_path: str = "experiments/checkpoints/dqn_agent.pth",
    lr: float = 1e-3,
    gamma: float = 0.9,
    epsilon_decay: float = 0.997,
    batch_size: int = 64,
    target_update_freq: int = 500,
) -> dict[str, list]:
    env = SnakeEnv(seed=seed)
    agent = DQNAgent(
        gamma=gamma,
        lr=lr,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
    )

    all_scores: list[int] = []
    avg_scores: list[float] = []
    epsilon_values: list[float] = []
    loss_values: list[float] = []

    for ep in range(1, episodes + 1):
        env.reset()
        state = env.get_state()
        episode_losses: list[float] = []

        for _ in range(max_steps):
            action = agent.choose_action(state)
            _, reward, done = env.step(action)
            next_state = env.get_state()

            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            if done:
                break

        all_scores.append(env.score)
        epsilon_values.append(agent.epsilon)
        loss_values.append(float(np.mean(episode_losses)) if episode_losses else 0.0)
        agent.decay_epsilon()

        if ep % 100 == 0:
            avg_scores.append(float(np.mean(all_scores[-100:])))

        if ep % 500 == 0:
            mean_last_100 = float(np.mean(all_scores[-100:]))
            print(
                f"Episode {ep}/{episodes} | Mean(100): {mean_last_100:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | Device: {agent.device}"
            )

    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    agent.save(save_path)

    mean_last_100 = avg_scores[-1] if avg_scores else float(np.mean(all_scores[-100:])) if all_scores else 0.0
    max_score = max(all_scores) if all_scores else 0
    print(f"Model saved to {save_path}")
    print(f"Training complete. Mean last 100: {mean_last_100:.2f}, Max: {max_score}")

    return {
        "scores": all_scores,
        "avg_scores": avg_scores,
        "epsilons": epsilon_values,
        "losses": loss_values,
    }
