"""RL training script — trains the DQN agent on procedurally generated levels."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RL_MAX_STEPS, RL_BATCH_SIZE, RL_TARGET_UPDATE, DEFAULT_GRID_SIZE
from game.engine import GameState
from game.level_generator import LevelGenerator
from solvers.rl_solver import RLSolver


def train(
    episodes: int = 2000,
    grid_size: int = DEFAULT_GRID_SIZE,
    difficulty: int = 1,
    save_dir: str = "models",
    save_every: int = 100,
    log_every: int = 10,
) -> None:
    """Train DQN agent on randomly generated levels."""
    os.makedirs(save_dir, exist_ok=True)

    agent = RLSolver(grid_size=grid_size)
    generator = LevelGenerator()

    # Try to resume training
    model_path = os.path.join(save_dir, f"dqn_grid{grid_size}_d{difficulty}.pt")
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Resuming from episode {agent.episodes_done}")

    print(f"Training DQN on {grid_size}x{grid_size} grid, difficulty {difficulty}")
    print(f"Device: {agent.device}")
    print(f"Episodes: {episodes}")
    print("-" * 60)

    rewards_history = []
    solved_history = []
    loss_history = []

    start_time = time.time()

    for ep in range(episodes):
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

        # Update target network
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

            print(
                f"Ep {agent.episodes_done:5d} | "
                f"Reward: {avg_reward:7.1f} | "
                f"Solved: {solve_rate:5.1f}% | "
                f"Loss: {avg_loss:.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Time: {elapsed:.0f}s"
            )

        # Save checkpoint
        if (ep + 1) % save_every == 0:
            agent.save(model_path)

    # Final save
    agent.save(model_path)
    print(f"\nTraining complete. Model saved to {model_path}")

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Smooth rewards
        window = min(50, len(rewards_history) // 4 + 1)
        smoothed = np.convolve(rewards_history, np.ones(window) / window, mode="valid")
        ax1.plot(smoothed, color="#ff6432")
        ax1.set_title("Episode Reward")
        ax1.set_xlabel("Episode")

        # Solve rate
        window = min(50, len(solved_history) // 4 + 1)
        smoothed_solved = np.convolve(solved_history, np.ones(window) / window, mode="valid") * 100
        ax2.plot(smoothed_solved, color="#64ff64")
        ax2.set_title("Solve Rate (%)")
        ax2.set_xlabel("Episode")

        # Loss
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
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Grid size")
    parser.add_argument("--difficulty", type=int, default=1, help="Difficulty (1-5)")
    parser.add_argument("--save-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N episodes")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        grid_size=args.grid_size,
        difficulty=args.difficulty,
        save_dir=args.save_dir,
        save_every=args.save_every,
    )
