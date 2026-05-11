"""Training loop for tabular Q-learning on the Snake environment.

This module contains pure training logic that runs episodes, updates a tabular
Q-learning agent, tracks learning metrics, and optionally saves the Q-table.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from .agent import DoubleQAgent, TabularQAgent
from .env import SnakeEnv


def train(
    episodes: int = 2000,
    width: int = 10,
    height: int = 10,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.997,
    max_steps_per_episode: int = 1000,
    log_every: int = 100,
    seed: int = 42,
    save_path: Optional[Union[str, "Path"]] = None,
    use_double: bool = False,
) -> Dict[str, Union[List[int], List[float]]]:
    env = SnakeEnv(width=width, height=height, seed=seed)
    agent = (
        DoubleQAgent(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )
        if use_double
        else TabularQAgent(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )
    )

    scores: List[int] = []
    lengths: List[int] = []
    epsilons: List[float] = []
    q_table_sizes: List[int] = []

    for episode in range(episodes):
        env.reset()
        done: bool = False
        steps: int = 0

        while not done and steps < max_steps_per_episode:
            state = env.get_state()
            action = agent.choose_action(state)
            _, reward, done = env.step(action)
            next_state = env.get_state()
            agent.update(state, action, reward, next_state, done)
            steps += 1

        scores.append(env.score)
        lengths.append(steps)
        epsilons.append(agent.epsilon)
        q_table_sizes.append(agent.q_table_size)

        agent.decay_epsilon()

        if log_every > 0 and (episode + 1) % log_every == 0:
            last_100 = scores[-100:]
            avg_score_last_100: float = sum(last_100) / len(last_100)
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Score: {env.score} | "
                f"AvgScore(Last100): {avg_score_last_100:.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"QStates: {agent.q_table_size}"
            )

    if save_path is not None:
        agent.save(save_path)

    max_score: int = max(scores) if scores else 0
    last_100_scores: List[int] = scores[-100:]
    mean_last_100: float = (
        sum(last_100_scores) / len(last_100_scores) if last_100_scores else 0.0
    )
    print(
        f"Training complete | Episodes: {episodes} | "
        f"MaxScore: {max_score} | "
        f"MeanLast100: {mean_last_100:.2f} | "
        f"FinalEpsilon: {agent.epsilon:.4f} | "
        f"StatesVisited: {agent.q_table_size}"
    )

    return {
        "scores": scores,
        "lengths": lengths,
        "epsilons": epsilons,
        "q_table_sizes": q_table_sizes,
    }
