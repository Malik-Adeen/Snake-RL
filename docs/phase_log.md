# Phase Log

Running record of decisions, changes, and rationale for each development phase.

---

## Phase 0 — Foundation ✅

**Goal:** Build a clean, bug-free base project from scratch.

**Files created:**
- `snake_rl/env.py` — SnakeEnv with relative actions, reward shaping hooks
- `snake_rl/agent.py` — TabularQAgent with save/load
- `snake_rl/train.py` — training loop returning metrics dict
- `snake_rl/renderer.py` — pygame renderer with parameterised grid size
- `snake_rl/__init__.py` — package exports
- `scripts/run_training.py` — CLI entry point with argparse + 3 plots
- `scripts/run_demo.py` — pygame demo with pause/quit controls
- `scripts/run_compare.py` — statistical comparison vs random agent
- `requirements.txt`

**Key decisions:**
- Renderer takes `grid_width` and `grid_height` as constructor parameters
  (previous version hardcoded 20, causing a size mismatch with the 10×10 env).
- Q-table saved via `agent.save()` at end of training
  (previous version never persisted the Q-table).
- `epsilon` parameter is passed through to the agent constructor
  (previous version silently hardcoded `epsilon=1.0` regardless of input).
- `max_steps_per_episode=1000` instead of 500 to allow longer episodes.
- `run_compare.py` loads the Q-table and sets `epsilon=0.0` before evaluation.

---

## Phase 1 — Evaluation & State Extension ✅

**Goal:** Understand the baseline agent's failure modes and extend the state.

### 1a — First training run (Run 1)

**Finding:** Agent learned to survive (mean ep length: 2000 = max cap) but not
to eat. Spinning in a loop was a rational strategy under the −0.1 step penalty.
Score mean: 1.74, max: 11. Q-table saturated at 64/128 states.

### 1b — Reward shaping (Run 2)

**Change:** Added distance-based step reward (±0.5) and progressive loop
penalty (−0.5 per 100 steps without food, capped at −3.0).

**Result:** Mean +24%, max 11→15. Agent now commits to food. State ceiling
unchanged at 64 states.

**Rationale for reward values chosen:**
- `+0.5 / -0.5` distance reward: small enough that food reward (+10) dominates,
  large enough to create a clear gradient.
- Loop penalty cap at −3.0: prevents the penalty from overwhelming the food
  signal at very long episodes.

### 1c — 8-bit state extension (Runs 3 & 4)

**Change:** Added `food_close = int(manhattan_dist <= 3)` as an 8th state bit.

**Rationale:** The agent had no information about distance to food — only
direction. Adding a binary "close" signal gives the agent a "commit mode"
context, splitting each of the 64 old states into two (close/far).

**Result:** States visited doubled 64→128. Max score unchanged at 15. Mean
score nearly identical after correcting for training budget (5000 episodes).

**Key finding:** Tripling training from 2000→5000 episodes produced no
measurable improvement once the Q-table was saturated. The state representation
is the binding constraint, not training duration or sample count.

---

## Phase 3 — Double Q-Learning ✅ Complete

**Goal:** Reduce Q-value overestimation to produce a more stable, consistent
policy within the same state space.

### Why Double Q-Learning?

Standard Q-learning computes the TD target as:

```
target = reward + γ · max_a Q(s', a)
```

The `max` operator introduces an upward bias — the same table is used to both
*select* the best action and *evaluate* it. This systematically overestimates
Q-values, making the agent overconfident in actions it hasn't fully explored.

Double Q-Learning maintains **two Q-tables** (A and B). At each update, one
table selects the action and the other evaluates it:

```
a* = argmax_a Q_A(s', a)          # table A selects
target = reward + γ · Q_B(s', a*) # table B evaluates
```

The role of selector and evaluator alternates randomly between tables.

**Expected outcome:** More conservative, less noisy Q-values. Policy variance
should decrease — fewer near-zero episodes, more consistent mid-range scores.
The absolute ceiling (max score ≈ 15) is unlikely to change since it is bounded
by the state representation, not the algorithm.

**Implementation plan:**
- Add `DoubleQAgent` class to `snake_rl/agent.py` alongside `TabularQAgent`
- Maintain two defaultdict Q-tables: `q_a` and `q_b`
- `choose_action`: uses the average of both tables for action selection
- `update`: randomly assigns update role to table A or B each step
- `save/load`: persists both tables
- `run_training.py`: add `--double` flag to switch agent class

**Files modified:** `snake_rl/agent.py`, `snake_rl/train.py`, `scripts/run_training.py`

**Results (Run 5, 5000 episodes):**
- Mean last-100: 2.34 (+29% vs standard Q at same config)
- Mean last-500: 1.96 (+17%)
- Max score: 14 (standard Q: 15 — marginal drop expected, overestimation bias removed)
- States visited: 128 (unchanged — algorithm does not affect reachable state space)

**Key finding:** Consistent improvement in average policy quality with no
architectural cost. Max score ceiling is set by the state representation,
not the algorithm — Double Q-Learning improves the floor, not the ceiling.

---

## Phase 2 — Visualisation ✅ Complete

All visualisation components built across Phases 0, 1, 3, and the demo polish sprint.

**Completed:**
- `snake_rl/renderer.py` — SnakeRenderer + MultiSnakeRenderer
  - Rounded stitched body segments with gradient coloring (bright head → dark tail)
  - Directional eyes on head (2 white circles with pupils, face movement direction)
  - Pulsing food with white specular highlight
  - HUD separator line
  - `render_to_surface()` method for subsurface compositing
- `scripts/run_demo.py` — single-agent demo
  - SPACE pause, ↑/↓ speed, Q quit, --epsilon, --max_steps, --double flags
  - `--show_qvalues` flag: live Q-value overlay on 3 adjacent cells
    Semi-transparent colored fill + border + arrow + value label + FWD/LEFT/RIGHT tag
    Green = best action, amber = middle, red = worst
- `scripts/run_heatmap.py` — static 4-direction policy heatmap saved as PNG
- `scripts/run_sidebyside.py` — random vs trained in one wide window
  - Two SnakeEnv instances with same episode seed (fair comparison)
  - White vertical divider, colour-coded HUD (red=random, green=trained)
- `scripts/run_human_vs_ai.py` — human (arrow keys) vs trained AI
  - 3-second countdown per episode
  - Result banner: YOU WIN / AI WINS / DRAW
  - = / - keys for speed (arrow keys reserved for direction)
- `scripts/record_gif.py` — captures gameplay frames and saves animated GIF
  (uses pillow, already in requirements)

---

## Phase 4 — Multi-Agent Extension ✅ Complete

Two-snake competitive Snake on a shared 10×10 board. Simultaneous actions,
collision resolution, separate rewards, shared food.

**Files created:**
- `snake_rl/multi_env.py` — MultiSnakeEnv (16-bit state per agent)
- `snake_rl/multi_train.py` — multi_train() with trained_vs_random and self_play modes
- `snake_rl/renderer.py` — MultiSnakeRenderer added (green = A, blue = B)
- `scripts/run_multi_training.py` — training entry point with score + win rate plots
- `scripts/run_multi_demo.py` — live visual demo, speed controls, optional model_b

**State per agent (16 bits):**
- Bits 0–2: danger_straight, danger_left, danger_right
- Bits 3–6: food_left, food_right, food_up, food_down
- Bit 7: food_close
- Bits 8–11: tail_left, tail_right, tail_up, tail_down
- Bits 12–15: opp_left, opp_right, opp_up, opp_down (opponent head)

**Training Mode 1 — trained_vs_random (5000 episodes):**

| Metric | Agent A (trained) | Agent B (random) |
|---|---|---|
| Mean score last 100 | 3.90 | 0.08 |
| Max score | 14 | 2 |
| Overall win rate A | 86.52% | — |

- Win rate crosses 50% at episode ~300
- Stabilises at 93–95% by episode 3000
- Random agent maxed at 2 across all 5000 episodes

**Training Mode 2 — self_play (5000 episodes):**

| Metric | Agent A | Agent B |
|---|---|---|
| Mean score last 100 | 2.85 | 3.64 |
| Max score | 11 | 18 |
| Overall win rate A | 39.82% | — |

- Neither agent dominates — win rate oscillates 40–45%, never breaks 50%
- Both agents improve together, scores tracking in parallel
- Agent B edges ahead — likely due to starting position advantage (right-center)
- Draws appear frequently in live demo (3/10 episodes)

**Key academic finding:**
> Agent A wins 86.52% against random but only 39.82% against a self-play
> trained opponent. This demonstrates that random-agent win rate is a poor
> measure of true policy quality. Self-play produces more robust agents
> that cannot be evaluated against a fixed weak baseline.

**Demo results (10 episodes each):**
- Trained A vs Random B: A wins 10/10, A mean 3.8, B mean 0.1
- Trained A vs Trained B: A 4 wins, B 3 wins, 3 draws — genuinely competitive

---

## Phase 5 — Academic Polish 🔄 In Progress

**Completed so far:**
- Multi-seed evaluation (Run 9): 3 seeds × 15000 episodes — mean ± std for all metrics
  Results: mean score 6.63 ± 0.12, max score 23 ± 1 — algorithm confirmed stable
- `README.md` at project root with GIF, headline results, quickstart, full experiment table
- `assets/demo_qoverlay.gif` — recorded via `record_gif.py`
- `tests/test_core.py` — 27 pytest tests covering env, agent, Q-update math, save/load,
  multi-agent env (27/27 pass in 0.28s)
- `pytest.ini` — test configuration
- Metric inconsistency resolved: experiment log now distinguishes training mean
  (6.50, seed=42) from evaluation mean (11.21, seed=99, frozen policy)
- Lazy pygame import in `snake_rl/__init__.py` — non-visual scripts now work
  in environments without pygame installed
- `.gitignore` — excludes __pycache__, .venv, generated plots; keeps models and assets
- Project pushed to GitHub

**DQN added (Run 10):**
- `snake_rl/dqn_agent.py` — QNetwork (12->128->128->3), ReplayBuffer, DQNAgent
- `snake_rl/dqn_train.py` — dqn_train() training loop
- `scripts/run_dqn_training.py` — entry point with training + eval + 3 plots
- Hardware: NVIDIA RTX 3060 Ti, CUDA 12.4 (PyTorch 2.5.1+cu124)
- Result: eval mean 11.42, max 25 — matches tabular in 5x fewer episodes (3000 vs 15000)
- Key finding: sample efficiency is the DQN advantage on this problem size;
  both methods share the same ~score 11 ceiling set by the 12-bit state abstraction

**Still to do:**
- Formal written report (Word document)
- Report-ready comparison plots (learning curves side by side, DQN vs tabular bar chart)
- References section (minimum 8-10 sources)
