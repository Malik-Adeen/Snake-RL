# Architecture — Snake RL

## 1. Project Overview

This project trains autonomous Snake agents using **Reinforcement Learning**,
progressing from Tabular Q-Learning through Double Q-Learning to a Deep Q-Network
(DQN) comparison. The goal is to produce a well-documented, reproducible RL
experiment that demonstrates both the capabilities and the hard limits of tabular
methods, then quantifies what function approximation adds.

Tabular Q-Learning is the primary focus because:

- The state space is small enough for an exact lookup table.
- The algorithm is fully interpretable — every learned value can be inspected.
- It serves as a meaningful baseline before the function-approximation step.
- It is well-suited to an academic setting where understanding *why* the agent
  acts is as important as *how well* it acts.

DQN is added in Phase 5 as a direct comparison using the identical environment
and state representation, isolating function approximation as the only variable.

---

## 2. Directory Structure

```
Snake-RL/
├── snake_rl/                   # Core package
│   ├── __init__.py             # Public API — lazy pygame import
│   ├── env.py                  # SnakeEnv — single-agent environment
│   ├── agent.py                # TabularQAgent + DoubleQAgent
│   ├── train.py                # train() — single-agent training loop
│   ├── dqn_agent.py            # QNetwork, ReplayBuffer, DQNAgent
│   ├── dqn_train.py            # dqn_train() — DQN training loop
│   ├── multi_env.py            # MultiSnakeEnv — two-snake competitive env
│   ├── multi_train.py          # multi_train() — multi-agent training loop
│   └── renderer.py             # SnakeRenderer + MultiSnakeRenderer
├── scripts/                    # Entry-point scripts
│   ├── run_training.py         # Train single agent, save plots + Q-table
│   ├── run_demo.py             # Watch agent play (+ --show_qvalues overlay)
│   ├── run_compare.py          # Trained vs random statistical comparison
│   ├── run_heatmap.py          # Policy heatmap visualisation (4 directions)
│   ├── run_sidebyside.py       # Random vs trained in split-screen window
│   ├── run_human_vs_ai.py      # Human (arrow keys) vs trained AI
│   ├── run_multi_training.py   # Train multi-agent, save plots + Q-tables
│   ├── run_multi_demo.py       # Watch two snakes compete live
│   ├── run_multiseed_eval.py   # Run N seeds, report mean ± std
│   ├── run_dqn_training.py     # Train DQN, evaluate, save 3 plots
│   └── record_gif.py           # Capture gameplay as animated GIF
├── experiments/
│   ├── checkpoints/            # Saved Q-tables (.pkl) and DQN weights (.pth)
│   ├── plots/                  # Generated training plots (.png)
│   └── multiseed_results.json  # Multi-seed evaluation output
├── tests/
│   ├── __init__.py
│   └── test_core.py            # 42 pytest tests: env, agent, Q-update, save/load,
│                               #   multi-agent env, QNetwork, ReplayBuffer, DQNAgent
├── docs/
│   ├── architecture.md         # This file
│   ├── experiment_log.md       # All 9 experiment runs with results
│   └── phase_log.md            # Development phase notes and decisions
├── assets/
│   └── demo_qoverlay.gif       # Gameplay GIF for README
├── README.md                   # Project front page with results and quickstart
├── pytest.ini                  # Pytest configuration
├── .gitignore
└── requirements.txt
```

---

## 3. Environment — `SnakeEnv`

| Property | Detail |
|---|---|
| Board | N × M grid (default 10 × 10) |
| Actions | 3 relative: straight (0), turn left (1), turn right (2) |
| Episode start | Snake of length 3, centered, facing right |
| Termination | Wall collision or self-collision |
| Food | One item; respawns on a random empty cell after collection |

### Reward shaping

| Event | Reward |
|---|---|
| Food eaten | +10.0 |
| Death | −10.0 |
| Step closer to food | +0.5 |
| Step further from food | −0.5 |
| Every 100 steps without food | −0.5 × (steps_without_food ÷ 100), capped at −3.0 |

The step-level reward shaping provides a dense gradient toward food and a
progressive penalty that discourages looping behaviour — the main failure mode
of agents trained with sparse rewards only.

---

## 4. Agents

### `TabularQAgent`

Standard epsilon-greedy Q-Learning. Stores a single Q-table as a `defaultdict`
mapping state tuples to a list of three floats (one per action).

**Q-Learning update rule:**
```
target = reward + γ · max_a Q(s', a) · (1 − done)
Q(s, a) ← Q(s, a) + α · (target − Q(s, a))
```

### `DoubleQAgent`

Reduces overestimation bias by maintaining two Q-tables (`q_a`, `q_b`).
At each update step, one table selects the best action and the other evaluates
it — the roles are assigned randomly 50/50:

```
a* = argmax_a Q_A(s', a)           # table A selects
target = reward + γ · Q_B(s', a*)  # table B evaluates
```

Action selection uses the **sum** of both tables for a combined estimate.
Produces more conservative, less noisy Q-values and a more consistent policy.

### Shared parameters

| Parameter | Default |
|---|---|
| α (alpha) | 0.1 |
| γ (gamma) | 0.9 |
| ε start | 1.0 |
| ε min | 0.05 |
| ε decay | 0.995 (final config) |

### Persistence

Both agents implement `save(path)` and `load(path)` via pickle.
`DoubleQAgent` saves `{"q_a": ..., "q_b": ...}` — incompatible with
`TabularQAgent` pickle format. Use `--double` flag in scripts to load correctly.

---

## 5. State Representation

### Single-agent state (12 bits)

| Index | Feature | Description |
|---|---|---|
| 0 | `danger_straight` | 1 if next cell ahead causes collision |
| 1 | `danger_left` | 1 if next cell left causes collision |
| 2 | `danger_right` | 1 if next cell right causes collision |
| 3 | `food_left` | 1 if food is left of head |
| 4 | `food_right` | 1 if food is right of head |
| 5 | `food_up` | 1 if food is above head |
| 6 | `food_down` | 1 if food is below head |
| 7 | `food_close` | 1 if Manhattan distance to food ≤ 3 |
| 8 | `tail_left` | 1 if own tail is left of head |
| 9 | `tail_right` | 1 if own tail is right of head |
| 10 | `tail_up` | 1 if own tail is above head |
| 11 | `tail_down` | 1 if own tail is below head |

**Theoretical maximum:** 2¹² = 4096 states
**Empirically reachable:** ~1017 on a 10 × 10 board

### Multi-agent state (16 bits per agent)

Same 12 bits as above, plus 4 opponent bits:

| Index | Feature | Description |
|---|---|---|
| 12 | `opp_left` | 1 if opponent head is left of own head |
| 13 | `opp_right` | 1 if opponent head is right of own head |
| 14 | `opp_up` | 1 if opponent head is above own head |
| 15 | `opp_down` | 1 if opponent head is below own head |

### State evolution across phases

| Phase | Bits added | Reason |
|---|---|---|
| 0 | 7 bits (baseline) | Danger + food direction |
| 1 | +1 `food_close` | Commit-mode signal for nearby food |
| 3 | +4 tail bits | Avoid self-trapping at longer lengths |
| 4 | +4 opponent bits | Opponent awareness in multi-agent env |

---

## 6. Training Pipeline

### Single-agent
```
for each episode:
    env.reset()
    while not done and steps < max_steps:
        state  = env.get_state()
        action = agent.choose_action(state)      # ε-greedy
        _, reward, done = env.step(action)
        next_state = env.get_state()
        agent.update(state, action, reward, next_state, done)
        steps += 1
    agent.decay_epsilon()

if save_path:
    agent.save(save_path)
```

### Multi-agent
```
for each episode:
    env.reset()
    while not env.done and steps < max_steps:
        state_a, state_b = env.get_state_a(), env.get_state_b()
        action_a = agent_a.choose_action(state_a)
        action_b = agent_b.choose_action(state_b)  # or random
        _, r_a, r_b, done_a, done_b, _ = env.step(action_a, action_b)
        agent_a.update(state_a, action_a, r_a, next_state_a, done_a)
        if self_play: agent_b.update(...)
        steps += 1
    agent_a.decay_epsilon()
```

### Final hyperparameters (best config — Run 6)

| Parameter | Value |
|---|---|
| Episodes | 15000 |
| Board size | 10 × 10 |
| α (alpha) | 0.1 |
| γ (gamma) | 0.9 |
| ε start | 1.0 |
| ε min | 0.05 |
| ε decay | 0.995 |
| Max steps/episode | 1000 |
| Agent | DoubleQAgent |
| State bits | 12 |

---

## 7. Multi-Agent Environment — `MultiSnakeEnv`

Two snakes compete on a shared board. Actions are resolved simultaneously
each step. Collision resolution order:

1. Head-on (both heads meet) → both die, −10 each
2. Snake A hits wall / own body / B's body → A dies, −10
3. Snake B hits wall / own body / A's body → B dies, −10
4. Food reached → collector gets +10, food respawns (A has priority if tie)
5. Step reward → ±0.5 based on distance delta to food

Episode ends when **both** snakes are dead. Each agent receives its own
separate reward signal, enabling independent Q-value learning.

---

## 8. Known Limitations

- **Tabular scalability.** Larger boards exponentially increase the required
  state space, making tabular methods impractical beyond simple configurations.
- **Binary state encoding.** All features are 0/1 — no continuous values.
  The agent cannot distinguish magnitudes, only directions and proximity.
- **Starting position asymmetry** in multi-agent mode. Agent B (right-center)
  has a slight advantage over Agent A (left-center) — observable in self-play
  results where B consistently outscores A.
- **No lookahead.** The agent reacts to the immediate next step only. Path
  planning around complex body configurations is beyond the state's expressive
  power.

---

## 9. Phase Roadmap

| Phase | Name | Status |
|---|---|---|
| 0 | Foundation | ✅ Complete |
| 1 | Evaluation & State Extension | ✅ Complete |
| 2 | Visualisation & Policy Heatmap | ✅ Complete |
| 3 | Double Q-Learning + 12-bit State | ✅ Complete |
| 4 | Multi-Agent Extension | ✅ Complete |
| 5 | Academic Polish & Report | 🔄 In progress |
