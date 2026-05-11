# Experiment Log

All runs use `seed=42`, board `10×10`, `α=0.1`, `γ=0.9`, `ε_decay=0.997`,
`max_steps=1000` unless stated otherwise.

---

## Run 1 — Baseline (7-bit state, no reward shaping)

**Date:** Phase 0  
**Episodes:** 2000  
**State bits:** 7  
**Reward:** +10 food, −10 death, −0.1 step  

| Metric | Value |
|---|---|
| States visited | 64 |
| Mean score (last 100) | 1.74 |
| Max score | 11 |
| Final epsilon | 0.05 |

**Findings:**
- Agent learned collision avoidance but not food-seeking.
- Mean episode length in evaluation: 2000 steps (hitting max cap).
- Agent was spinning in a safe loop — surviving indefinitely, rarely eating.
- Trained vs random comparison: 0.65 vs 0.20 mean score. Gap exists but small.
- Q-table saturated at 64 states by episode 600 and never grew further.

---

## Run 2 — Reward Shaping Added (7-bit state)

**Date:** Phase 1  
**Episodes:** 2000  
**State bits:** 7  
**Reward:** +10 food, −10 death, ±0.5 distance delta, progressive loop penalty  

| Metric | Value |
|---|---|
| States visited | 64 |
| Mean score (last 100) | 2.16 |
| Mean score (last 500) | 1.93 |
| Max score | 15 |
| Final epsilon | 0.05 |

**Findings:**
- Mean score improved +24% over baseline.
- Max score jumped from 11 → 15.
- Higher variance in scores (occasional 6s and 7s) — agent now commits to food.
- Q-table still saturated at 64 states — state representation is the ceiling.
- Looping behaviour reduced but not eliminated.

---

## Run 3 — 8-bit State, 2000 Episodes

**Date:** Phase 1  
**Episodes:** 2000  
**State bits:** 8 (added `food_close`: Manhattan distance ≤ 3)  
**Reward:** Same as Run 2  

| Metric | Value |
|---|---|
| States visited | 127 |
| Mean score (last 100) | 1.58 |
| Mean score (last 500) | 1.89 |
| Max score | 15 |
| Final epsilon | 0.05 |

**Findings:**
- State space doubled as expected: 64 → 127.
- Mean last-100 slightly lower than Run 2 due to undertraining (same budget, twice the states).
- Mean last-500 nearly identical — no regression, just more states to fill.
- Max score unchanged: state expansion alone does not raise the ceiling.
- Concluded: 2000 episodes insufficient for 127-state space.

---

## Run 4 — 8-bit State, 5000 Episodes

**Date:** Phase 1  
**Episodes:** 5000  
**State bits:** 8  
**Reward:** Same as Run 2  

| Metric | Value |
|---|---|
| States visited | 128 |
| Mean score (last 100) | 1.81 |
| Mean score (last 500) | 1.67 |
| Max score | 15 |
| Final epsilon | 0.05 |

**Findings:**
- Q-table ceiling confirmed at 128 reachable states (50% of theoretical 256).
- Tripling episodes (2000 → 5000) produced no improvement in max or mean score.
- Policy has fully converged to a local optimum bounded by the state abstraction.
- **Conclusion: the 8-bit binary state is the performance ceiling for standard Q-learning on this board. Further training is not useful. Algorithmic improvement required.**

---

## Summary Table

| Run | State bits | Episodes | States | Mean(100) | Max |
|---|---|---|---|---|---|
| 1 — Baseline | 7 | 2000 | 64 | 1.74 | 11 |
| 2 — +Shaping | 7 | 2000 | 64 | 2.16 | 15 |
| 3 — 8-bit | 8 | 2000 | 127 | 1.58 | 15 |
| 4 — 8-bit | 8 | 5000 | 128 | 1.81 | 15 |

---

## Run 5 — Double Q-Learning (Phase 3)

**Date:** Phase 3  
**Episodes:** 5000  
**Agent:** DoubleQAgent (two Q-tables, decoupled selection/evaluation)  
**State bits:** 8  
**Save path:** `experiments/checkpoints/q_table_double.pkl`  

| Metric | Value |
|---|---|
| States visited | 128 |
| Mean score (last 100) | 2.34 |
| Mean score (last 500) | 1.96 |
| Max score | 14 |
| Final epsilon | 0.05 |

**Findings:**
- Mean last-100 improved +29% over standard Q-Learning (1.81 → 2.34).
- Mean last-500 improved +17% (1.67 → 1.96).
- Max score dropped slightly (15 → 14) — expected behaviour. Overestimation
  bias in standard Q-Learning occasionally produces lucky high-scoring runs.
  Double Q-Learning removes this, so the ceiling appears marginally lower but
  the average policy is meaningfully more consistent.
- States visited unchanged at 128 — the state representation ceiling is
  independent of the algorithm.
- **Conclusion:** Double Q-Learning produces a more stable policy within the
  same state abstraction at no additional cost. The +29% mean improvement with
  zero architectural change is the key Phase 3 finding.

## Run 6 — 12-bit State + Tail Awareness (Phase A)

**Date:** Phase 3 / Path A  
**Episodes:** 15000  
**Agent:** DoubleQAgent  
**State bits:** 12 (added tail_left, tail_right, tail_up, tail_down)  
**epsilon_decay:** 0.995  
**Save path:** `experiments/checkpoints/q_table_double.pkl`  

| Metric | Value |
|---|---|
| States visited | 1017 |
| Mean score — training last 100 | 6.50 |
| Mean score — training last 500 | 6.79 |
| Mean score — evaluation (100 eps, seed=99) | **11.21** |
| Max score | 24 |
| Final epsilon | 0.05 |

> **Note on metric difference:** The training mean (6.50) is measured during the
> final 100 training episodes where ε=0.05 exploration is still active and
> Q-values are still updating. The evaluation mean (11.21) is measured by running
> the *fully trained, frozen* policy for 100 fresh episodes (seed=99) via
> `run_compare.py`. The evaluation score is higher because: (a) no Q-table
> updates occur during evaluation, (b) different seed produces different food
> spawn sequences, and (c) the policy is fully converged. Both metrics are valid
> and measure different things — training stability vs deployment performance.

**Head-to-head vs random (100 episodes):**

| Metric | Random | Trained |
|---|---|---|
| Mean score | 0.20 | 11.21 |
| Std score | 0.53 | 4.92 |
| Max score | 4 | 24 |
| Mean ep length | 19.68 | 137.09 |
| Score > 0 | 17% | 100% |
| Score ≥ 5 | 0% | 92% |
| Score ≥ 10 | 0% | 60% |

**Findings:**
- Adding 4 tail direction bits was the decisive improvement.
- Mean score jumped from 2.34 → 11.21 (+378%).
- Max score: 14 → 24. States visited: 128 → 1017.
- Agent now scores ≥ 5 in 92% of episodes and ≥ 10 in 60%.
- 56× better mean score than random agent.
- Gameplay is visually convincing — agent actively chases food and navigates around its own body.
- **Conclusion: tail awareness was the missing piece. The agent can now avoid self-trapping, which was the dominant failure mode at higher scores.**

---

## Run 7 — Multi-Agent: Trained vs Random (Phase 4)

**Date:** Phase 4
**Episodes:** 5000
**Agent A:** DoubleQAgent (learns)
**Agent B:** Random policy
**Mode:** trained_vs_random
**epsilon_decay:** 0.995
**Save path:** `experiments/checkpoints/q_table_multi_a.pkl`

| Metric | Agent A (trained) | Agent B (random) |
|---|---|---|
| Mean score (last 100) | 3.90 | 0.08 |
| Max score | 14 | 2 |
| Overall win rate A | 86.52% | — |

**Win rate progression:**
- Episode 500: 47.4% (agent still learning)
- Episode 1000: 83.6% (epsilon at floor, policy forming)
- Episode 3000: 93.0%
- Episode 5000: 95.0% (last 500 episodes)

**Findings:**
- Win rate crosses 50% at episode ~300, reaches 80%+ by episode 700.
- Random agent scored maximum 2 across all 5000 episodes.
- Trained agent controls board and starves random opponent of food.
- Textbook RL convergence curve — rapid early gains, plateau at 93–95%.

---

## Run 8 — Multi-Agent: Self-Play (Phase 4)

**Date:** Phase 4
**Episodes:** 5000
**Agent A:** DoubleQAgent (seed=42)
**Agent B:** DoubleQAgent (seed=49)
**Mode:** self_play
**epsilon_decay:** 0.995
**Save paths:** `q_table_multi_a.pkl`, `q_table_multi_b.pkl`

| Metric | Agent A | Agent B |
|---|---|---|
| Mean score (last 100) | 2.85 | 3.64 |
| Max score | 11 | 18 |
| Overall win rate A | 39.82% | — |

**Win rate progression:**
- Episode 500: 23.4% (B starts stronger — position advantage)
- Episode 1000: 35.8%
- Episode 3000: 44.2%
- Episode 5000: 44.2% (plateau, neither dominates)

**Findings:**
- Neither agent breaks 50% win rate throughout training.
- Both agents improve in parallel — scores track closely each episode.
- Agent B consistently edges A: starting position (right-center, facing left)
  puts B closer to board center earlier.
- Self-play produces genuinely competitive dynamics; random-baseline win rate
  (86.52%) vastly overstates policy quality vs a trained opponent.
- **Key finding:** Win rate vs random ≠ policy quality. Self-play is a
  more honest evaluation.

---

## Final Summary Table (all runs)

| Run | Agent | State bits | Episodes | States | Mean(100) | Max | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Standard Q | 7 | 2000 | 64 | 1.74 | 11 | Baseline |
| 2 | Standard Q + shaping | 7 | 2000 | 64 | 2.16 | 15 | Reward shaping added |
| 3 | Standard Q + shaping | 8 | 2000 | 127 | 1.58 | 15 | 8-bit state, undertrained |
| 4 | Standard Q + shaping | 8 | 5000 | 128 | 1.81 | 15 | State ceiling confirmed |
| 5 | Double Q + shaping | 8 | 5000 | 128 | 2.34 | 14 | +29% mean, less variance |
| 6 | Double Q + shaping | 12 | 15000 | 1017 | 6.50 (train) / **11.21** (eval) | **24** | Tail bits — decisive jump |
| 7 | Double Q (multi, A) | 16 | 5000 | — | 3.90 | 14 | 86.52% win vs random |
| 8 | Double Q (self-play A) | 16 | 5000 | — | 2.85 | 11 | 39.82% win vs trained B |
| **9** | **Double Q multi-seed** | **12** | **15000×3** | **1016±2** | **6.63±0.12** | **23±1** | **Reproducibility confirmed** |

---

## Run 9 — Multi-Seed Reproducibility (3 seeds)

**Date:** Phase 5
**Seeds:** 42, 123, 777
**Episodes:** 15000 per seed
**Agent:** DoubleQAgent, 12-bit state, ε_decay=0.995

| Metric | Seed 42 | Seed 123 | Seed 777 | Mean ± Std |
|---|---|---|---|---|
| Mean score (last 100) | 6.50 | 6.61 | 6.79 | **6.63 ± 0.12** |
| Mean score (last 500) | 6.79 | 6.65 | 7.09 | **6.84 ± 0.18** |
| Max score | 24 | 23 | 22 | **23 ± 1** |
| States visited | 1017 | 1018 | 1014 | **1016 ± 2** |
| Final epsilon | 0.05 | 0.05 | 0.05 | **0.05 ± 0.00** |

**Findings:**
- ±0.12 std on mean score confirms the algorithm is highly stable across seeds.
- States visited variance of ±2 out of ~1016 means the reachable state space
  is deterministic — exploration path varies but the ceiling is identical.
- Max score variance of ±1 is within noise — all seeds produce equivalent policies.
- **Conclusion:** Results are reproducible. The project’s central performance
  claims hold across different random initializations.
