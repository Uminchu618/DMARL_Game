"""CoinGameEnv をランダム方策で数ステップ動かし、簡易に可視化するスクリプト。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from rl_lab.envs import CoinGameEnv


def run_episode(steps: int, render_mode: str | None) -> None:
    env = CoinGameEnv(render_mode=render_mode)
    obs, info = env.reset(seed=0)
    print(f"initial step={info['step']} reward_vec={info['rewards']}")
    if render_mode == "ansi":
        print(env.render())

    for t in range(1, steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"t={t:02d} act={action} reward={reward:.2f} "
            f"vec={info['rewards']} trunc={truncated}"
        )
        if render_mode == "ansi":
            print(env.render())
        if terminated or truncated:
            break

    if render_mode == "rgb_array":
        frame = env.render()
        if frame is not None:
            out = Path("examples") / "coin_game_last_frame.npy"
            np.save(out, frame)
            print(f"Saved last rgb frame to {out} (shape={frame.shape})")

    env.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Coin Game quick demo")
    parser.add_argument(
        "--steps", type=int, default=5, help="実行ステップ数"
    )
    parser.add_argument(
        "--render-mode",
        choices=[None, "ansi", "rgb_array"],
        default="ansi",
        help="render_mode を選択。ansi ならターミナル描画、rgb_array なら numpy を保存",
    )
    args = parser.parse_args(argv)
    run_episode(steps=args.steps, render_mode=args.render_mode)


if __name__ == "__main__":
    main()
