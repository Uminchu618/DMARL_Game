import numpy as np

from rl_lab.envs import CoinGameEnv, CoinState


def test_reset_returns_observation_and_info():
    env = CoinGameEnv(grid_size=8, coins_per_color=2, max_steps=5)
    observation, info = env.reset(seed=123)

    assert observation.shape == (8, 8, 3)
    assert observation.dtype == np.float32
    assert np.all(observation >= 0.0) and np.all(observation <= 1.0)

    rewards = info["rewards"]
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (2,)
    assert np.allclose(rewards, 0.0)


def test_collecting_own_color_rewards_agent():
    env = CoinGameEnv(grid_size=4, coins_per_color=1, max_steps=5)
    env.reset(seed=0)
    env.agent_positions = np.array([(0, 0), (3, 3)], dtype=np.int64)
    env.coins = [
        CoinState(owner=0, position=(0, 1)),
        CoinState(owner=1, position=(2, 2)),
    ]

    _, reward, terminated, truncated, info = env.step([4, 0])

    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    np.testing.assert_allclose(info["rewards"], np.array([1.0, 0.0]))


def test_collecting_opponent_coin_penalizes_opponent():
    env = CoinGameEnv(grid_size=4, coins_per_color=1, max_steps=5)
    env.reset(seed=0)
    env.agent_positions = np.array([(0, 0), (1, 1)], dtype=np.int64)
    env.coins = [
        CoinState(owner=1, position=(0, 1)),
        CoinState(owner=0, position=(3, 3)),
    ]

    _, reward, _, _, info = env.step([4, 0])

    assert reward == 2.0
    np.testing.assert_allclose(info["rewards"], np.array([2.0, -3.0]))
