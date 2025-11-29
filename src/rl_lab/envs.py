from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


@dataclass
class CoinState:
    """コインの状態を保持するデータ構造。

    Attributes:
        owner: コインの所有エージェントID（0始まり）。
        position: グリッド上の(y, x)座標。
    """

    owner: int
    position: Tuple[int, int]


class CoinGameEnv(gym.Env[np.ndarray, np.ndarray]):
    """Coin Game を gymnasium.Env と互換の API で提供する環境クラス。

    観測は (grid_size, grid_size, 3) 形状の float32 テンソル（0〜1）で、
    以下のチャンネル構造を持つ：

    - チャンネル0: エージェント0とその色のコインを赤チャンネルとして表現（コインは0.5、エージェントは1.0）
    - チャンネル1: エージェント1とその色のコインを青チャンネルとして表現（コインは0.5、エージェントは1.0）
    - チャンネル2: 予備チャンネル（現在は0で埋める）

    報酬は PD 的な構造に基づく：自身のコインを取ると +1、相手のコインを取ると +2（相手には -3）を与える。
    返り値の reward には controlled_agent_index のエージェントの報酬を返し、
    info["rewards"] には全エージェント分のベクトルを格納する。
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 10}

    _ACTION_TO_DELTA = {
        0: (0, 0),
        1: (-1, 0),  # up
        2: (1, 0),   # down
        3: (0, -1),  # left
        4: (0, 1),   # right
    }

    def __init__(
        self,
        grid_size: int = 32,
        coins_per_color: int = 10,
        max_steps: int = 100,
        render_mode: str | None = None,
        controlled_agent_index: int = 0,
    ) -> None:
        if grid_size <= 0:
            raise ValueError("grid_size は 1 以上にしてください。")
        if coins_per_color <= 0:
            raise ValueError("coins_per_color は 1 以上にしてください。")
        if max_steps <= 0:
            raise ValueError("max_steps は 1 以上にしてください。")
        if render_mode not in (None, "rgb_array", "ansi"):
            raise ValueError("render_mode は None, 'rgb_array', 'ansi' のいずれかです。")

        # 本実装は2エージェント（協調構造の PD を想定）に固定する。
        self.num_agents = 2
        if not 0 <= controlled_agent_index < self.num_agents:
            raise ValueError("controlled_agent_index はエージェント数の範囲内で指定してください。")

        self.grid_size = grid_size
        self.coins_per_color = coins_per_color
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.controlled_agent_index = controlled_agent_index

        self.action_space = spaces.MultiDiscrete(np.full(self.num_agents, len(self._ACTION_TO_DELTA), dtype=np.int64))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.float32,
        )

        self.agent_positions: np.ndarray | None = None
        self.coins: List[CoinState] = []
        self.step_count = 0
        self._last_observation: np.ndarray | None = None
        self._np_random = None
        self._np_random_seed = None

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """環境を初期化し、最初の観測と info を返す。

        Args:
            seed: 乱数シード。指定しない場合は内部でサンプリングする。
            options: 予備のオプション（未使用）。

        Returns:
            tuple[np.ndarray, dict[str, Any]]: 初期観測と補助情報。
                観測は (grid_size, grid_size, 3) 形状の float32 テンソル。
        """
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        elif self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random(None)

        self.step_count = 0
        self.agent_positions = self._sample_initial_positions()
        self.coins = []
        for owner in range(self.num_agents):
            for _ in range(self.coins_per_color):
                self._spawn_coin(owner)

        observation = self._build_observation()
        info = self._build_info(np.zeros(self.num_agents, dtype=np.float32))
        self._last_observation = observation
        return observation, info

    def step(
        self, action: Sequence[int] | np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """行動を適用して1ステップ進める。

        Args:
            action: 各エージェントの行動を表す配列。shape は (2,) 固定。
                0: 停止, 1: 上, 2: 下, 3: 左, 4: 右。

        Returns:
            observation: 次状態の観測。shape は (grid_size, grid_size, 3)。
            reward: controlled_agent_index に対応するエージェントの報酬。
            terminated: ゲーム終了フラグ（本環境では常に False）。
            truncated: ステップ上限で打ち切られたかどうか。
            info: rewards ベクトルや座標などの補助情報。
        """
        if self.agent_positions is None:
            raise RuntimeError("reset() を呼び出した後に step() を実行してください。")

        actions = self._validate_action(action)
        self._apply_actions(actions)
        reward_vector = self._collect_coins()

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        observation = self._build_observation()
        info = self._build_info(reward_vector)
        self._last_observation = observation
        controlled_reward = float(reward_vector[self.controlled_agent_index])
        return observation, controlled_reward, terminated, truncated, info

    def render(self) -> np.ndarray | str | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            if self._last_observation is None:
                return self._build_observation()
            return np.copy(self._last_observation)
        if self.render_mode == "ansi":
            return self._ascii_render()
        return None

    def close(self) -> None:  # noqa: D401
        """追加のリソースは無いので何もしない。"""
        return

    # --- 内部ユーティリティ ---

    def _validate_action(self, action: Sequence[int] | np.ndarray) -> np.ndarray:
        array = np.asarray(action, dtype=np.int64)
        if array.shape == ():
            array = np.array([int(array)])
        if array.shape != (self.num_agents,):
            raise ValueError(f"action は shape {(self.num_agents,)} で指定してください。")
        if not self.action_space.contains(array):
            raise ValueError("action が action_space に含まれていません。")
        return array

    def _sample_initial_positions(self) -> np.ndarray:
        positions = []
        occupied: set[Tuple[int, int]] = set()
        for _ in range(self.num_agents):
            pos = self._sample_empty_cell(occupied)
            occupied.add(pos)
            positions.append(pos)
        return np.asarray(positions, dtype=np.int64)

    def _sample_empty_cell(self, occupied: set[Tuple[int, int]]) -> Tuple[int, int]:
        while True:
            y = int(self._np_random.integers(0, self.grid_size))
            x = int(self._np_random.integers(0, self.grid_size))
            if (y, x) not in occupied:
                return (y, x)

    def _spawn_coin(self, owner: int) -> None:
        occupied = {tuple(pos) for pos in self.agent_positions} | {coin.position for coin in self.coins}
        pos = self._sample_empty_cell(occupied)
        self.coins.append(CoinState(owner=owner, position=pos))

    def _apply_actions(self, actions: np.ndarray) -> None:
        assert self.agent_positions is not None
        for idx, act in enumerate(actions):
            dy, dx = self._ACTION_TO_DELTA[int(act)]
            y, x = self.agent_positions[idx]
            ny = int(np.clip(y + dy, 0, self.grid_size - 1))
            nx = int(np.clip(x + dx, 0, self.grid_size - 1))
            self.agent_positions[idx] = (ny, nx)

    def _collect_coins(self) -> np.ndarray:
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        remaining: List[CoinState] = []
        occupied_by_agent = {tuple(pos): idx for idx, pos in enumerate(self.agent_positions)}

        for coin in self.coins:
            agent_idx = occupied_by_agent.get(coin.position)
            if agent_idx is None:
                remaining.append(coin)
                continue

            if agent_idx == coin.owner:
                rewards[agent_idx] += 1.0
            else:
                rewards[agent_idx] += 2.0
                rewards[coin.owner] -= 3.0
            self._spawn_coin(coin.owner)

        self.coins = remaining
        return rewards

    def _build_observation(self) -> np.ndarray:
        canvas = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        if self.agent_positions is None:
            return canvas

        # コインの可視化
        for coin in self.coins:
            y, x = coin.position
            channel = coin.owner if coin.owner < 2 else coin.owner % 2
            canvas[y, x, channel] = max(canvas[y, x, channel], 0.5)

        # エージェントの可視化（上書き優先）
        for idx, (y, x) in enumerate(self.agent_positions):
            channel = idx if idx < 2 else idx % 2
            canvas[y, x, channel] = 1.0

        return canvas

    def _ascii_render(self) -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for coin in self.coins:
            y, x = coin.position
            grid[y][x] = "r" if coin.owner == 0 else "b"
        positions = [] if self.agent_positions is None else self.agent_positions
        for idx, (y, x) in enumerate(positions):
            grid[y][x] = "A" if idx == 0 else "B"
        return "\n".join("".join(row) for row in grid)

    def _build_info(self, rewards: np.ndarray) -> Dict[str, Any]:
        return {
            "rewards": rewards.copy(),
            "agent_positions": None if self.agent_positions is None else self.agent_positions.copy(),
            "coins": [(coin.owner, coin.position) for coin in self.coins],
            "step": self.step_count,
            "seed": self._np_random_seed,
        }


__all__ = ["CoinGameEnv", "CoinState"]
