"""Focused RL training — 10x10 grids with traps only.

Goal: train an agent that can solve ANY 10x10 maze with traps,
as long as a solution exists.

Key differences from general training:
  - No keys, doors, or enemies — just navigate + avoid traps
  - Larger network (more filters)
  - Longer training (50k+ episodes)
  - Curriculum on trap density only
  - Higher reward for avoiding traps (+5 for each step without trap hit)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RL_MAX_STEPS, RL_TARGET_UPDATE
from game.level_generator import LevelGenerator
from solvers.rl_solver import RLSolver

# Curriculum: (grid_size, num_traps, label)
# Start tiny, build up to full 10x10 with traps
TRAP_CURRICULUM = [
    (5, 0, "5x5 no traps"),
    (7, 0, "7x7 no traps"),
    (10, 0, "10x10 no traps"),
    (10, 3, "10x10 + 3 traps"),
    (10, 8, "10x10 + 8 traps"),
    (10, 15, "10x10 + 15 traps"),
    (10, 25, "10x10 + 25 traps"),
]

ADVANCE_THRESHOLD = 0.40
ADVANCE_WINDOW = 100
MAX_GRID_SIZE = 10


def train_traps(
    episodes: int = 50000,
    save_dir: str = "models",
    snapshot_every: int = 2000,
    log_every: int = 50,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    snapshot_dir = os.path.join(save_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    agent = RLSolver(grid_size=MAX_GRID_SIZE)
    generator = LevelGenerator()

    model_path = os.path.join(save_dir, f"dqn_traps_grid{MAX_GRID_SIZE}.pt")
    stage = 0
    if os.path.exists(model_path):
        checkpoint = agent.load(model_path)
        stage = checkpoint.get("stage", 0)
        print(f"Resuming from episode {agent.episodes_done}, stage {stage}")

    grid_size, num_traps, label = TRAP_CURRICULUM[stage]

    print(f"Training DQN — TRAPS ONLY (max grid {MAX_GRID_SIZE}x{MAX_GRID_SIZE})")
    print(f"Device: {agent.device}")
    print(f"Episodes: {episodes}")
    print(f"Curriculum: {len(TRAP_CURRICULUM)} stages, advance at {ADVANCE_THRESHOLD*100:.0f}% over {ADVANCE_WINDOW} episodes")
    print(f"Stage 0: {label}")
    print("-" * 60)

    rewards_history = []
    solved_history = []
    stage_solved = []
    loss_history = []
    start_time = time.time()

    for ep in range(episodes):
        # Generate level — traps only, no keys/doors/enemies
        level = generator.generate(
            size=grid_size,
            num_keys=0,
            num_traps=num_traps,
            num_enemies=0,
            difficulty=1,
        )

        state = level.copy()
        episode_reward = 0.0
        losses = []

        for step in range(RL_MAX_STEPS):
            obs = agent._pad_observation(state)
            action = agent.select_action(state, training=True)
            next_state, reward, done = state.step(action)
            next_obs = agent._pad_observation(next_state)

            agent.memory.push(obs, action, reward, next_obs, done)
            episode_reward += reward

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            if done:
                break

        agent.decay_epsilon()
        won = 1 if state.won else 0
        rewards_history.append(episode_reward)
        solved_history.append(won)
        stage_solved.append(won)

        if losses:
            loss_history.append(np.mean(losses))

        if (ep + 1) % RL_TARGET_UPDATE == 0:
            agent.update_target()

        # Logging
        if (ep + 1) % log_every == 0:
            recent_r = rewards_history[-log_every:]
            recent_s = solved_history[-log_every:]
            elapsed = time.time() - start_time
            print(
                f"Ep {agent.episodes_done:5d} [S{stage} {grid_size}x{grid_size} {num_traps}t] | "
                f"Reward: {np.mean(recent_r):7.1f} | "
                f"Solved: {np.mean(recent_s)*100:5.1f}% | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Time: {elapsed:.0f}s"
            )

        # Curriculum advancement
        if len(stage_solved) >= ADVANCE_WINDOW:
            recent_rate = np.mean(stage_solved[-ADVANCE_WINDOW:])
            if recent_rate >= ADVANCE_THRESHOLD and stage < len(TRAP_CURRICULUM) - 1:
                snap_path = os.path.join(
                    snapshot_dir,
                    f"dqn_traps_grid{MAX_GRID_SIZE}_stage{stage}_ep{agent.episodes_done}.pt",
                )
                agent.save(snap_path, stage=stage)

                stage += 1
                grid_size, num_traps, label = TRAP_CURRICULUM[stage]
                stage_solved = []
                print(f"\n>>> ADVANCING to Stage {stage}: {label} "
                      f"(solve rate was {recent_rate*100:.1f}%)\n")

        # Save checkpoint
        if (ep + 1) % 500 == 0:
            agent.save(model_path, stage=stage)

        # Save snapshot
        if (ep + 1) % snapshot_every == 0:
            snap_path = os.path.join(
                snapshot_dir,
                f"dqn_traps_grid{MAX_GRID_SIZE}_ep{agent.episodes_done}.pt",
            )
            agent.save(snap_path)

    agent.save(model_path, stage=stage)
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s. Model saved to {model_path}")
    print(f"Reached stage {stage}/{len(TRAP_CURRICULUM)-1}: {TRAP_CURRICULUM[stage][1]}")
    print(f"Final solve rate (last {log_every}): {np.mean(solved_history[-log_every:])*100:.1f}%")

    # Plot
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        window = min(100, len(rewards_history) // 4 + 1)
        smoothed_r = np.convolve(rewards_history, np.ones(window) / window, mode="valid")
        ax1.plot(smoothed_r, color="#ff6432")
        ax1.set_title("Episode Reward")
        ax1.set_xlabel("Episode")

        smoothed_s = np.convolve(solved_history, np.ones(window) / window, mode="valid") * 100
        ax2.plot(smoothed_s, color="#64ff64")
        ax2.set_title("Solve Rate (%)")
        ax2.set_xlabel("Episode")

        plt.suptitle(f"DQN Traps Training — {GRID_SIZE}x{GRID_SIZE}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_traps_curves.png"), dpi=150)
        plt.show()
    except Exception as e:
        print(f"Could not plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on 10x10 trap-only levels")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory for model and snapshots (e.g. /content/drive/MyDrive/models)")
    parser.add_argument("--snapshot-every", type=int, default=2000)
    args = parser.parse_args()

    train_traps(
        episodes=args.episodes,
        save_dir=args.save_dir,
        snapshot_every=args.snapshot_every,
    )
