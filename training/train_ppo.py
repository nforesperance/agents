"""PPO training script — trains with stable-baselines3.

Usage:
    python training/train_ppo.py --steps 500000 --grid-size 10
    python training/train_ppo.py --steps 500000 --grid-size 10 --traps 10
    python training/train_ppo.py --steps 500000 --save-dir /content/drive/MyDrive/ai_models
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from config import DEFAULT_GRID_SIZE
from solvers.ppo_solver import PuzzleEnv


class ProgressCallback(BaseCallback):
    """Logs solve rate and saves snapshots during training."""

    def __init__(
        self,
        eval_env,
        save_dir: str,
        grid_size: int,
        num_traps: int,
        snapshot_every: int = 50000,
        eval_every: int = 10000,
        eval_episodes: int = 50,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.grid_size = grid_size
        self.num_traps = num_traps
        self.snapshot_every = snapshot_every
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.start_time = time.time()
        self.best_solve_rate = 0.0
        os.makedirs(os.path.join(save_dir, "snapshots"), exist_ok=True)

    def _on_step(self) -> bool:
        step = self.num_timesteps

        if step % self.eval_every == 0 and step > 0:
            wins = 0
            total_reward = 0.0
            for _ in range(self.eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    done = done or truncated
                total_reward += ep_reward
                if info.get("won", False):
                    wins += 1

            solve_rate = wins / self.eval_episodes
            avg_reward = total_reward / self.eval_episodes
            elapsed = time.time() - self.start_time

            print(
                f"Step {step:>8d} | "
                f"Solved: {solve_rate*100:5.1f}% | "
                f"Reward: {avg_reward:7.1f} | "
                f"Time: {elapsed:.0f}s"
            )

            if solve_rate > self.best_solve_rate:
                self.best_solve_rate = solve_rate
                best_path = os.path.join(self.save_dir, f"ppo_grid{self.grid_size}_best.zip")
                self.model.save(best_path)
                print(f"  New best! ({solve_rate*100:.1f}%) saved to {best_path}")

        if step % self.snapshot_every == 0 and step > 0:
            snap_path = os.path.join(
                self.save_dir, "snapshots",
                f"ppo_grid{self.grid_size}_t{self.num_traps}_step{step}.zip",
            )
            self.model.save(snap_path)
            print(f"  Snapshot saved: {snap_path}")

        return True


def make_env(grid_size, num_traps):
    def _init():
        return PuzzleEnv(grid_size=grid_size, num_traps=num_traps)
    return _init


def train_ppo(
    total_steps: int = 500_000,
    grid_size: int = 10,
    num_traps: int = 0,
    save_dir: str = "models",
    n_envs: int = 8,
    snapshot_every: int = 50000,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"ppo_grid{grid_size}_t{num_traps}.zip")

    print(f"PPO Training — {grid_size}x{grid_size} grid, {num_traps} traps")
    print(f"Total steps: {total_steps:,}")
    print(f"Parallel envs: {n_envs}")
    print("-" * 60)

    # Parallel training environments
    env = DummyVecEnv([make_env(grid_size, num_traps) for _ in range(n_envs)])

    # Eval environment (single)
    eval_env = PuzzleEnv(grid_size=grid_size, num_traps=num_traps)

    # Create or load model
    if os.path.exists(model_path):
        print(f"Resuming from {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
        )

    callback = ProgressCallback(
        eval_env=eval_env,
        save_dir=save_dir,
        grid_size=grid_size,
        num_traps=num_traps,
        snapshot_every=snapshot_every,
    )

    start = time.time()
    model.learn(total_timesteps=total_steps, callback=callback)
    elapsed = time.time() - start

    model.save(model_path)
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Model saved to {model_path}")
    print(f"Best solve rate: {callback.best_solve_rate*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for puzzle solving")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    parser.add_argument("--traps", type=int, default=0, help="Number of traps")
    parser.add_argument("--save-dir", type=str, default="models", help="Save directory")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--snapshot-every", type=int, default=50000, help="Snapshot interval")
    args = parser.parse_args()

    train_ppo(
        total_steps=args.steps,
        grid_size=args.grid_size,
        num_traps=args.traps,
        save_dir=args.save_dir,
        n_envs=args.n_envs,
        snapshot_every=args.snapshot_every,
    )
