"""Training loop for two-agent competitive Snake in MultiSnakeEnv.

This module provides a training routine for competitive play with either a
learning agent against a random opponent or simultaneous self-play learning.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Union

from .agent import DoubleQAgent, TabularQAgent
from .multi_env import MultiSnakeEnv


def multi_train(
    episodes: int = 5000,
    width: int = 10,
    height: int = 10,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_decay: float = 0.995,
    max_steps: int = 1000,
    log_every: int = 500,
    seed: int = 42,
    mode: str = "trained_vs_random",
    save_path_a: Optional[str] = None,
    save_path_b: Optional[str] = None,
) -> Dict[str, List]:
    if mode not in ("trained_vs_random", "self_play"):
        raise ValueError("mode must be either 'trained_vs_random' or 'self_play'")

    env: MultiSnakeEnv = MultiSnakeEnv(width=width, height=height, seed=seed)
    random_opponent = random.Random(seed + 1)

    agent_a: Union[DoubleQAgent, TabularQAgent] = DoubleQAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    agent_b: Optional[Union[DoubleQAgent, TabularQAgent]] = None
    if mode == "self_play":
        agent_b = DoubleQAgent(
            alpha=alpha,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            seed=seed + 7,
        )

    scores_a: List[int] = []
    scores_b: List[int] = []
    lengths: List[int] = []

    for episode in range(episodes):
        env.reset()
        steps: int = 0

        while not env.done and steps < max_steps:
            state_a = env.get_state_a()
            state_b = env.get_state_b()

            action_a: int = agent_a.choose_action(state_a)
            if mode == "trained_vs_random":
                action_b: int = random_opponent.choice([0, 1, 2])
            else:
                if agent_b is None:
                    raise RuntimeError("agent_b is required for self_play mode")
                action_b = agent_b.choose_action(state_b)

            _, reward_a, reward_b, done_a, done_b, episode_done = env.step(action_a, action_b)

            next_state_a = env.get_state_a()
            next_state_b = env.get_state_b()

            agent_a.update(state_a, action_a, reward_a, next_state_a, done_a)
            if mode == "self_play":
                if agent_b is None:
                    raise RuntimeError("agent_b is required for self_play mode")
                agent_b.update(state_b, action_b, reward_b, next_state_b, done_b)

            steps += 1
            if episode_done:
                break

        scores_a.append(env.score_a)
        scores_b.append(env.score_b)
        lengths.append(steps)

        agent_a.decay_epsilon()
        if mode == "self_play":
            if agent_b is None:
                raise RuntimeError("agent_b is required for self_play mode")
            agent_b.decay_epsilon()

        if log_every > 0 and (episode + 1) % log_every == 0:
            last_n: int = min(log_every, episode + 1)
            recent_a: List[int] = scores_a[-last_n:]
            recent_b: List[int] = scores_b[-last_n:]
            avg_a: float = sum(recent_a) / float(last_n)
            avg_b: float = sum(recent_b) / float(last_n)
            wins_a: int = sum(1 for a_score, b_score in zip(recent_a, recent_b) if a_score > b_score)
            win_rate_a: float = 100.0 * wins_a / float(last_n)
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"AvgScore_A: {avg_a:.2f} | "
                f"AvgScore_B: {avg_b:.2f} | "
                f"WinRate_A: {win_rate_a:.1f}% | "
                f"Epsilon: {agent_a.epsilon:.4f} | "
                f"Mode: {mode}"
            )

    if save_path_a:
        path_a = Path(save_path_a)
        agent_a.save(path_a)
    if save_path_b and mode == "self_play":
        if agent_b is None:
            raise RuntimeError("agent_b is required for self_play mode")
        path_b = Path(save_path_b)
        agent_b.save(path_b)

    max_score_a: int = max(scores_a) if scores_a else 0
    max_score_b: int = max(scores_b) if scores_b else 0
    last_100_a: List[int] = scores_a[-100:]
    last_100_b: List[int] = scores_b[-100:]
    mean_last_100_a: float = sum(last_100_a) / float(len(last_100_a)) if last_100_a else 0.0
    mean_last_100_b: float = sum(last_100_b) / float(len(last_100_b)) if last_100_b else 0.0
    overall_wins_a: int = sum(1 for a_score, b_score in zip(scores_a, scores_b) if a_score > b_score)
    overall_win_rate_a: float = (
        (overall_wins_a / float(len(scores_a))) * 100.0 if scores_a else 0.0
    )
    print(
        f"Training complete | Episodes: {episodes} | "
        f"MaxScore_A: {max_score_a} | "
        f"MaxScore_B: {max_score_b} | "
        f"MeanLast100_A: {mean_last_100_a:.2f} | "
        f"MeanLast100_B: {mean_last_100_b:.2f} | "
        f"WinRate_A: {overall_win_rate_a:.1f}%"
    )

    return {
        "scores_a": scores_a,
        "scores_b": scores_b,
        "lengths": lengths,
        "win_rate_a": overall_win_rate_a,
    }
