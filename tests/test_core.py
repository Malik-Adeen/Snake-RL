"""Basic test suite for Snake RL core components.

Covers: environment transitions, collision rules,
Q-update math, agent save/load, multi-agent env.
"""

import tempfile
import pytest

from snake_rl.env import SnakeEnv
from snake_rl.agent import TabularQAgent, DoubleQAgent
from snake_rl.multi_env import MultiSnakeEnv


# ─────────────────────────────────────────────
# SnakeEnv
# ─────────────────────────────────────────────

class TestSnakeEnv:
    def test_reset_returns_snake_and_food(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        snake, food = env.reset()
        assert len(snake) == 3
        assert food not in snake

    def test_invalid_board_size_raises(self):
        with pytest.raises(ValueError):
            SnakeEnv(width=2, height=10)

    def test_step_on_done_raises(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        env.done = True
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_invalid_action_raises(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        with pytest.raises(ValueError):
            env.step(99)

    def test_food_reward_on_eat(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        head_x, head_y = env.snake[0]
        env.food = (head_x + 1, head_y)   # one step ahead (facing right)
        _, reward, done = env.step(0)
        assert reward == pytest.approx(10.0)
        assert not done
        assert env.score == 1

    def test_death_on_wall(self):
        env = SnakeEnv(width=5, height=5, seed=0)
        env.reset()
        env.snake = [(4, 2), (3, 2), (2, 2)]
        env.direction_idx = 0
        env.food = (0, 0)
        _, reward, done = env.step(0)
        assert reward == pytest.approx(-10.0)
        assert done

    def test_state_is_12_tuple_of_bits(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        state = env.get_state()
        assert len(state) == 12
        assert all(v in (0, 1) for v in state)

    def test_snake_grows_on_food(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        initial_length = len(env.snake)
        head_x, head_y = env.snake[0]
        env.food = (head_x + 1, head_y)
        env.step(0)
        assert len(env.snake) == initial_length + 1

    def test_snake_length_constant_on_step(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        length = len(env.snake)
        env.food = (0, 0)
        _, _, done = env.step(0)
        if not done:
            assert len(env.snake) == length


# ─────────────────────────────────────────────
# TabularQAgent
# ─────────────────────────────────────────────

class TestTabularQAgent:
    def _agent(self):
        return TabularQAgent(alpha=0.1, gamma=0.9, epsilon=1.0, seed=0)

    def test_choose_action_valid(self):
        agent = self._agent()
        s = (0,) * 12
        for _ in range(20):
            assert agent.choose_action(s) in (0, 1, 2)

    def test_bellman_update_exact(self):
        """alpha=1.0 so new Q = target exactly."""
        agent = TabularQAgent(alpha=1.0, gamma=0.5, epsilon=0.0, seed=0)
        s  = (0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0)
        s2 = (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0)
        agent.q_table[s2] = [4.0, 2.0, 0.0]
        agent.update(s, 0, 8.0, s2, False)
        # target = 8 + 0.5 * 4 = 10
        assert agent.q_table[s][0] == pytest.approx(10.0)

    def test_done_masks_future(self):
        agent = TabularQAgent(alpha=1.0, gamma=0.9, epsilon=0.0, seed=0)
        s  = (1,) + (0,) * 11
        s2 = (0,) * 12
        agent.q_table[s2] = [100.0, 100.0, 100.0]
        agent.update(s, 1, -10.0, s2, True)
        assert agent.q_table[s][1] == pytest.approx(-10.0)

    def test_epsilon_decay_floor(self):
        agent = TabularQAgent(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9, seed=0)
        for _ in range(200):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.05)

    def test_invalid_action_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError):
            agent.update((0,)*12, 5, 1.0, (0,)*12, False)

    def test_save_load_roundtrip(self):
        agent = TabularQAgent(seed=0)
        s = (1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1)
        agent.q_table[s] = [1.5, 2.5, 3.5]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        agent.save(path)
        agent2 = TabularQAgent(seed=0)
        agent2.load(path)
        assert agent2.q_table[s] == pytest.approx([1.5, 2.5, 3.5])

    def test_load_missing_raises(self):
        agent = TabularQAgent(seed=0)
        with pytest.raises(FileNotFoundError):
            agent.load("nonexistent_xyz.pkl")

    def test_q_table_size(self):
        agent = self._agent()
        assert agent.q_table_size == 0
        _ = agent.q_table[(0,)*12]
        assert agent.q_table_size == 1


# ─────────────────────────────────────────────
# DoubleQAgent
# ─────────────────────────────────────────────

class TestDoubleQAgent:
    def test_choose_action_valid(self):
        agent = DoubleQAgent(epsilon=0.0, seed=0)
        assert agent.choose_action((0,)*12) in (0, 1, 2)

    def test_update_modifies_exactly_one_table(self):
        agent = DoubleQAgent(alpha=1.0, gamma=0.0, epsilon=0.0, seed=0)
        s  = (1,) + (0,) * 11
        s2 = (0,) * 12
        a_before = list(agent.q_a[s])
        b_before = list(agent.q_b[s])
        agent.update(s, 0, 5.0, s2, False)
        a_changed = agent.q_a[s] != a_before
        b_changed = agent.q_b[s] != b_before
        assert a_changed != b_changed   # exactly one changed

    def test_save_load_roundtrip(self):
        agent = DoubleQAgent(seed=0)
        s = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1)
        agent.q_a[s] = [1.0, 2.0, 3.0]
        agent.q_b[s] = [4.0, 5.0, 6.0]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        agent.save(path)
        agent2 = DoubleQAgent(seed=0)
        agent2.load(path)
        assert agent2.q_a[s] == pytest.approx([1.0, 2.0, 3.0])
        assert agent2.q_b[s] == pytest.approx([4.0, 5.0, 6.0])

    def test_q_table_size_union(self):
        agent = DoubleQAgent(seed=0)
        sa = (1,) + (0,) * 11
        sb = (0, 1) + (0,) * 10
        agent.q_a[sa] = [0.0, 0.0, 0.0]
        agent.q_b[sb] = [0.0, 0.0, 0.0]
        assert agent.q_table_size == 2


# ─────────────────────────────────────────────
# MultiSnakeEnv
# ─────────────────────────────────────────────

class TestMultiSnakeEnv:
    def test_reset_no_overlap(self):
        env = MultiSnakeEnv(width=10, height=10, seed=0)
        env.reset()
        combined = list(env.snake_a) + list(env.snake_b)
        assert len(set(combined)) == len(combined)
        assert env.food not in env.snake_a
        assert env.food not in env.snake_b

    def test_invalid_board_raises(self):
        with pytest.raises(ValueError):
            MultiSnakeEnv(width=3, height=10)

    def test_step_on_done_raises(self):
        env = MultiSnakeEnv(width=10, height=10, seed=0)
        env.reset()
        env.done = True
        with pytest.raises(RuntimeError):
            env.step(0, 0)

    def test_state_a_is_16_bits(self):
        env = MultiSnakeEnv(width=10, height=10, seed=0)
        env.reset()
        s = env.get_state_a()
        assert len(s) == 16
        assert all(v in (0, 1) for v in s)

    def test_state_b_is_16_bits(self):
        env = MultiSnakeEnv(width=10, height=10, seed=0)
        env.reset()
        s = env.get_state_b()
        assert len(s) == 16
        assert all(v in (0, 1) for v in s)

    def test_dead_agent_state_is_zeros(self):
        env = MultiSnakeEnv(width=10, height=10, seed=0)
        env.reset()
        env.done_a = True
        env.snake_a = []
        assert all(v == 0 for v in env.get_state_a())
