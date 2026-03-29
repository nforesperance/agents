"""RL training script — trains the DQN agent with curriculum learning.

Curriculum stages (auto-advance when solve rate > threshold):
  Stage 0: 7x7 grid, no keys, no traps, no enemies — just reach the goal
  Stage 1: 7x7 grid, 1 trap, no enemies
  Stage 2: 7x7 grid, 2 traps, no enemies
  Stage 3: 7x7 grid, 1 key + door, 2 traps, no enemies
  Stage 4: 9x9 grid, 1 key, 3 traps, 1 enemy (= difficulty 1)
  Stage 5: 9x9 grid, difficulty 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RL_MAX_STEPS, RL_BATCH_SIZE, RL_TARGET_UPDATE, DEFAULT_GRID_SIZE
from game.engine import GameState
from game.level_generator import LevelGenerator
from solvers.rl_solver import RLSolver


# Curriculum stages: (grid_size, num_keys, num_traps, num_enemies, label)
CURRICULUM = [
    (7, 0, 0, 0, "7x7 goal only"),
    (7, 0, 1, 0, "7x7 + 1 trap"),
    (7, 0, 2, 0, "7x7 + 2 traps"),
    (7, 1, 2, 0, "7x7 + key/door + 2 traps"),
    (9, 1, 3, 1, "9x9 full (difficulty 1)"),
    (9, 2, 5, 2, "9x9 hard (difficulty 2)"),
]

ADVANCE_THRESHOLD = 0.6   # 60% solve rate to advance
ADVANCE_WINDOW = 200      # over the last 200 episodes


def train(
    episodes: int = 2000,
    grid_size: int = DEFAULT_GRID_SIZE,
    difficulty: int = 1,
    save_dir: str = "models",
    save_every: int = 100,
    snapshot_every: int = 500,
    log_every: int = 10,
    curriculum: bool = True,
) -> None:
    """Train DQN agent on procedurally generated levels.

    If curriculum=True, starts from simple levels and auto-advances.
    If curriculum=False, trains directly on the specified difficulty.
    """
    os.makedirs(save_dir, exist_ok=True)
    snapshot_dir = os.path.join(save_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    # Use max grid size from curriculum for the network
    max_grid = max(s[0] for s in CURRICULUM) if curriculum else grid_size
    agent = RLSolver(grid_size=max_grid)
    generator = LevelGenerator()

    model_path = os.path.join(save_dir, f"dqn_grid{max_grid}_d{difficulty}.pt")
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Resuming from episode {agent.episodes_done}")

    # Determine starting stage
    stage = 0
    if curriculum:
        print(f"Curriculum training with {len(CURRICULUM)} stages")
        print(f"Advance threshold: {ADVANCE_THRESHOLD*100:.0f}% over last {ADVANCE_WINDOW} episodes")
    else:
        print(f"Direct training on {grid_size}x{grid_size} grid, difficulty {difficulty}")

    print(f"Device: {agent.device}")
    print(f"Episodes: {episodes}")
    print("-" * 60)

    if curriculum:
        g, nk, nt, ne, label = CURRICULUM[stage]
        print(f"Stage {stage}: {label}")

    rewards_history = []
    solved_history = []
    loss_history = []

    start_time = time.time()

    for ep in range(episodes):
        # Generate level based on curriculum or fixed difficulty
        if curriculum:
            g, nk, nt, ne, label = CURRICULUM[stage]
            level = generator.generate(
                size=g, num_keys=nk, num_traps=nt, num_enemies=ne, difficulty=1,
            )
        else:
            level = generator.generate(size=grid_size, difficulty=difficulty)

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
        rewards_history.append(episode_reward)
        solved_history.append(1 if state.won else 0)

        if losses:
            loss_history.append(np.mean(losses))

        if (ep + 1) % RL_TARGET_UPDATE == 0:
            agent.update_target()

        # Logging
        if (ep + 1) % log_every == 0:
            recent_rewards = rewards_history[-log_every:]
            recent_solved = solved_history[-log_every:]
            avg_reward = np.mean(recent_rewards)
            solve_rate = np.mean(recent_solved) * 100
            avg_loss = np.mean(loss_history[-log_every:]) if loss_history else 0
            elapsed = time.time() - start_time

            stage_str = f"S{stage}" if curriculum else f"D{difficulty}"
            print(
                f"Ep {agent.episodes_done:5d} [{stage_str}] | "
                f"Reward: {avg_reward:7.1f} | "
                f"Solved: {solve_rate:5.1f}% | "
                f"Loss: {avg_loss:.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Time: {elapsed:.0f}s"
            )

        # Curriculum advancement
        if curriculum and len(solved_history) >= ADVANCE_WINDOW:
            recent_rate = np.mean(solved_history[-ADVANCE_WINDOW:])
            if recent_rate >= ADVANCE_THRESHOLD and stage < len(CURRICULUM) - 1:
                stage += 1
                g, nk, nt, ne, label = CURRICULUM[stage]
                print(f"\n>>> ADVANCING to Stage {stage}: {label} "
                      f"(solve rate was {recent_rate*100:.1f}%)\n")

        # Save checkpoint
        if (ep + 1) % save_every == 0:
            agent.save(model_path)

        # Save snapshot
        if (ep + 1) % snapshot_every == 0:
            snap_path = os.path.join(
                snapshot_dir,
                f"dqn_grid{max_grid}_d{difficulty}_ep{agent.episodes_done}.pt",
            )
            agent.save(snap_path)

    # Final save
    agent.save(model_path)
    print(f"\nTraining complete. Model saved to {model_path}")
    if curriculum:
        print(f"Reached stage {stage}/{len(CURRICULUM)-1}: {CURRICULUM[stage][4]}")

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        window = min(50, len(rewards_history) // 4 + 1)
        smoothed = np.convolve(rewards_history, np.ones(window) / window, mode="valid")
        ax1.plot(smoothed, color="#ff6432")
        ax1.set_title("Episode Reward")
        ax1.set_xlabel("Episode")

        window = min(50, len(solved_history) // 4 + 1)
        smoothed_solved = np.convolve(solved_history, np.ones(window) / window, mode="valid") * 100
        ax2.plot(smoothed_solved, color="#64ff64")
        ax2.set_title("Solve Rate (%)")
        ax2.set_xlabel("Episode")

        if loss_history:
            ax3.plot(loss_history, color="#00c8ff", alpha=0.5)
            ax3.set_title("Training Loss")
            ax3.set_xlabel("Episode")

        plt.suptitle("DQN Training Progress", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
        plt.show()
        print(f"Training curves saved to {save_dir}/training_curves.png")
    except Exception as e:
        print(f"Could not plot training curves: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for puzzle solving")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Grid size")
    parser.add_argument("--difficulty", type=int, default=1, help="Difficulty (1-5)")
    parser.add_argument("--save-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--snapshot-every", type=int, default=500, help="Save snapshot every N episodes")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        grid_size=args.grid_size,
        difficulty=args.difficulty,
        save_dir=args.save_dir,
        save_every=args.save_every,
        snapshot_every=args.snapshot_every,
        curriculum=not args.no_curriculum,
    )
