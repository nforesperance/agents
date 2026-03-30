"""PPO training script with curriculum — trains with stable-baselines3.

Curriculum: starts on 5x5, advances through 7x7, 9x9, 10x10, then adds traps.

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
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import DEFAULT_GRID_SIZE
from solvers.ppo_solver import PuzzleEnv

# Curriculum: (grid_size, num_traps, max_steps, label)
# max_steps scales with grid — small mazes don't need 200 steps
CURRICULUM = [
    (5, 0, 50, "5x5 no traps"),
    (7, 0, 80, "7x7 no traps"),
    (9, 0, 150, "9x9 no traps"),
    (10, 0, 200, "10x10 no traps"),
    (10, 5, 200, "10x10 + 5 traps"),
    (10, 10, 200, "10x10 + 10 traps"),
    (10, 20, 200, "10x10 + 20 traps"),
]

ADVANCE_THRESHOLD = 0.50
EVAL_EPISODES = 50
STEPS_PER_STAGE = 50_000


def evaluate(model, grid_size, num_traps, obs_size=10, max_steps=200, n_episodes=50):
    """Evaluate model on fresh levels."""
    env = PuzzleEnv(grid_size=grid_size, num_traps=num_traps, obs_size=obs_size, max_steps=max_steps)
    wins = 0
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            done = done or truncated
        total_reward += ep_reward
        if info.get("won", False):
            wins += 1
    return wins / n_episodes, total_reward / n_episodes


def make_env(grid_size, num_traps, obs_size=10, max_steps=200):
    def _init():
        return PuzzleEnv(grid_size=grid_size, num_traps=num_traps, obs_size=obs_size, max_steps=max_steps)
    return _init


def train_ppo(
    total_steps: int = 500_000,
    target_grid: int = 10,
    target_traps: int = 0,
    save_dir: str = "models",
    n_envs: int = 8,
    snapshot_every: int = 100_000,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    snap_dir = os.path.join(save_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"ppo_grid{target_grid}_t{target_traps}.zip")
    state_path = os.path.join(save_dir, f"ppo_grid{target_grid}_t{target_traps}_state.json")

    # Filter curriculum up to the target
    curriculum = [(g, t, ms, l) for g, t, ms, l in CURRICULUM if g <= target_grid and t <= target_traps]
    if not curriculum or curriculum[-1][:2] != (target_grid, target_traps):
        curriculum.append((target_grid, target_traps, 200, f"{target_grid}x{target_grid} + {target_traps} traps"))

    # Load state
    stage = 0
    steps_done = 0
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
            stage = state.get("stage", 0)
            steps_done = state.get("steps_done", 0)
        print(f"Resuming: stage {stage}, {steps_done:,} steps done")

    grid_size, num_traps, max_steps, label = curriculum[min(stage, len(curriculum) - 1)]

    print(f"PPO Curriculum Training — target {target_grid}x{target_grid} + {target_traps} traps")
    print(f"Total steps: {total_steps:,}  |  Stages: {len(curriculum)}")
    print(f"Parallel envs: {n_envs}  |  Device: cpu (MlpPolicy)")
    print(f"Stage {stage}: {label}")
    print("-" * 60)

    # Create environments for current stage
    env = DummyVecEnv([make_env(grid_size, num_traps, obs_size=target_grid, max_steps=max_steps) for _ in range(n_envs)])

    # Create or load model
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env, device="cpu")
        print(f"Loaded model from {model_path}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            device="cpu",
        )

    start_time = time.time()
    best_solve_rate = 0.0
    remaining = total_steps - steps_done

    while remaining > 0 and stage < len(curriculum):
        grid_size, num_traps, max_steps, label = curriculum[stage]

        # Update envs for current stage
        env = DummyVecEnv([make_env(grid_size, num_traps, obs_size=target_grid, max_steps=max_steps) for _ in range(n_envs)])
        model.set_env(env)

        # Train for a chunk
        chunk = min(STEPS_PER_STAGE, remaining)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        remaining -= chunk

        # Evaluate
        solve_rate, avg_reward = evaluate(model, grid_size, num_traps, obs_size=target_grid, max_steps=max_steps, n_episodes=EVAL_EPISODES)
        elapsed = time.time() - start_time

        print(
            f"Step {steps_done:>8,} [S{stage} {label}] | "
            f"Solved: {solve_rate*100:5.1f}% | "
            f"Reward: {avg_reward:7.1f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Save checkpoint
        model.save(model_path)
        with open(state_path, "w") as f:
            json.dump({"stage": stage, "steps_done": steps_done}, f)

        # Save best
        if solve_rate > best_solve_rate:
            best_solve_rate = solve_rate
            best_path = os.path.join(save_dir, f"ppo_grid{target_grid}_best.zip")
            model.save(best_path)
            print(f"  New best! {solve_rate*100:.1f}%")

        # Snapshot
        if steps_done % snapshot_every < STEPS_PER_STAGE:
            snap = os.path.join(snap_dir, f"ppo_grid{target_grid}_stage{stage}_step{steps_done}.zip")
            model.save(snap)

        # Advance curriculum
        if solve_rate >= ADVANCE_THRESHOLD and stage < len(curriculum) - 1:
            snap = os.path.join(snap_dir, f"ppo_grid{target_grid}_stage{stage}_done.zip")
            model.save(snap)
            stage += 1
            grid_size, num_traps, max_steps, label = curriculum[stage]
            print(f"\n>>> ADVANCING to Stage {stage}: {label} (was {solve_rate*100:.1f}%)\n")
            with open(state_path, "w") as f:
                json.dump({"stage": stage, "steps_done": steps_done}, f)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Best solve rate: {best_solve_rate*100:.1f}%")
    print(f"Final stage: {stage}/{len(curriculum)-1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent with curriculum")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--grid-size", type=int, default=10, help="Target grid size")
    parser.add_argument("--traps", type=int, default=0, help="Target number of traps")
    parser.add_argument("--save-dir", type=str, default="models", help="Save directory")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--snapshot-every", type=int, default=100_000, help="Snapshot interval")
    args = parser.parse_args()

    train_ppo(
        total_steps=args.steps,
        target_grid=args.grid_size,
        target_traps=args.traps,
        save_dir=args.save_dir,
        n_envs=args.n_envs,
        snapshot_every=args.snapshot_every,
    )
