"""CoinGameEnv をランダム方策で動かし、GIF 動画として保存するデモ。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
import numpy as np

from rl_lab.envs import CoinGameEnv


def run_and_save(steps: int, seed: int | None, output: Path, duration: float) -> None:
    env = CoinGameEnv(render_mode="rgb_array")
    frames = []

    obs, _ = env.reset(seed=seed)
    frames.append((obs * 255).astype(np.uint8))

    for _ in range(steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append((obs * 255).astype(np.uint8))
        if terminated or truncated:
            break

    env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, frames, duration=duration, loop=0)
    print(f"Saved GIF to {output} (frames={len(frames)}, duration/frame={duration}s)")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Coin Game GIF renderer")
    parser.add_argument("--steps", type=int, default=20, help="ステップ数")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--output", type=Path, default=Path("examples/coin_game.gif"), help="出力パス")
    parser.add_argument(
        "--duration", type=float, default=0.2, help="1フレームの表示秒数 (GIF 用)"
    )
    args = parser.parse_args(argv)
    run_and_save(steps=args.steps, seed=args.seed, output=args.output, duration=args.duration)


if __name__ == "__main__":
    main()
