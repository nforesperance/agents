"""RL solver — DQN agent using PyTorch."""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    UP, DOWN, LEFT, RIGHT, WAIT,
    RL_LEARNING_RATE, RL_GAMMA, RL_EPSILON_START, RL_EPSILON_END,
    RL_EPSILON_DECAY, RL_BATCH_SIZE, RL_MEMORY_SIZE, RL_TARGET_UPDATE,
    RL_MAX_STEPS, DEFAULT_GRID_SIZE,
)
from game.engine import GameState
from solvers.base import BaseSolver


class DQN(nn.Module):
    """Deep Q-Network with convolutional layers for grid input."""

    def __init__(self, in_channels: int = 5, grid_size: int = DEFAULT_GRID_SIZE, n_actions: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_size = 64 * grid_size * grid_size
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        features = self.conv(x).view(batch, -1)
        return self.fc(features)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = RL_MEMORY_SIZE):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class RLSolver(BaseSolver):
    """DQN-based RL solver.

    Can be used in two modes:
    - Training mode: learns from experience
    - Inference mode: loads trained weights and solves greedily
    """

    name = "RL (DQN)"
    color = "rl"

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.grid_size = grid_size
        self.n_actions = 5  # UP, DOWN, LEFT, RIGHT, WAIT

        self.policy_net = DQN(grid_size=grid_size, n_actions=self.n_actions).to(self.device)
        self.target_net = DQN(grid_size=grid_size, n_actions=self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=RL_LEARNING_RATE)
        self.memory = ReplayBuffer()
        self.epsilon = RL_EPSILON_START
        self.steps_done = 0
        self.episodes_done = 0

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def select_action(self, state: GameState, training: bool = False) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        obs = self._pad_observation(state)
        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def solve(self, state: GameState) -> list[int]:
        """Greedy inference — no exploration."""
        actions = []
        current = state.copy()
        for _ in range(RL_MAX_STEPS):
            if current.done:
                break
            action = self.select_action(current, training=False)
            actions.append(action)
            current, _, _ = current.step(action)
        return actions

    def train_step(self) -> Optional[float]:
        """One gradient step on a batch from replay buffer."""
        if len(self.memory) < RL_BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(RL_BATCH_SIZE)

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + RL_GAMMA * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(RL_EPSILON_END, self.epsilon * RL_EPSILON_DECAY)
        self.episodes_done += 1

    def _pad_observation(self, state: GameState) -> np.ndarray:
        """Pad/crop observation to fixed grid_size."""
        obs = state.to_observation()
        c, h, w = obs.shape
        padded = np.zeros((c, self.grid_size, self.grid_size), dtype=np.float32)
        ph = min(h, self.grid_size)
        pw = min(w, self.grid_size)
        padded[:, :ph, :pw] = obs[:, :ph, :pw]
        return padded

    def save(self, path: str, **extra) -> None:
        data = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episodes_done": self.episodes_done,
        }
        data.update(extra)
        torch.save(data, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", RL_EPSILON_END)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        print(f"Model loaded from {path} (episode {self.episodes_done}, epsilon {self.epsilon:.3f})")
        return checkpoint

    def reset(self) -> None:
        pass
