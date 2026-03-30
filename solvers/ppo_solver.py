"""PPO solver — uses Proximal Policy Optimization from stable-baselines3.

PPO learns a *policy* (strategy) rather than a value table like DQN,
making it much better at generalizing across random maze layouts.
"""

from __future__ import annotations

import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    UP, DOWN, LEFT, RIGHT, WAIT,
    FLOOR, WALL, TRAP, KEY, DOOR, GOAL, ENEMY, START,
    RL_MAX_STEPS, DEFAULT_GRID_SIZE,
)
from game.engine import GameState
from game.level_generator import LevelGenerator
from solvers.base import BaseSolver


class PuzzleEnv(gym.Env):
    """Gymnasium environment wrapper for the puzzle game."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        num_traps: int = 0,
        num_keys: int = 0,
        num_enemies: int = 0,
        max_steps: int = RL_MAX_STEPS,
        obs_size: int = 10,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.obs_size = obs_size  # fixed observation size for all stages
        self.num_traps = num_traps
        self.num_keys = num_keys
        self.num_enemies = num_enemies
        self.max_steps = max_steps
        self.generator = LevelGenerator()

        # Flattened observation: always padded to obs_size x obs_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(5 * obs_size * obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)  # UP, DOWN, LEFT, RIGHT, WAIT

        self.state: Optional[GameState] = None
        self.reset()

    def _pad_obs(self, state: GameState) -> np.ndarray:
        obs = state.to_observation()
        c, h, w = obs.shape
        padded = np.zeros((c, self.obs_size, self.obs_size), dtype=np.float32)
        ph, pw = min(h, self.obs_size), min(w, self.obs_size)
        padded[:, :ph, :pw] = obs[:, :ph, :pw]
        return padded.flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.generator.generate(
            size=self.grid_size,
            num_keys=self.num_keys,
            num_traps=self.num_traps,
            num_enemies=self.num_enemies,
            difficulty=1,
        )
        self.state.max_steps = self.max_steps
        return self._pad_obs(self.state), {}

    def step(self, action):
        self.state, reward, done = self.state.step(int(action))
        obs = self._pad_obs(self.state)
        truncated = False
        return obs, reward, done, truncated, {"won": self.state.won}


class PPOSolver(BaseSolver):
    """PPO-based solver using stable-baselines3."""

    name = "RL (PPO)"
    color = "rl"

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        model_path: Optional[str] = None,
    ):
        self.grid_size = grid_size
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def solve(self, state: GameState) -> list[int]:
        if self.model is None:
            # Random actions if no model loaded
            import random
            actions = []
            current = state.copy()
            for _ in range(RL_MAX_STEPS):
                if current.done:
                    break
                a = random.randrange(4)
                actions.append(a)
                current, _, _ = current.step(a)
            return actions

        actions = []
        current = state.copy()
        obs = self._pad_obs(current)

        for _ in range(RL_MAX_STEPS):
            if current.done:
                break
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(int(action))
            current, _, _ = current.step(int(action))
            obs = self._pad_obs(current)

        return actions

    def _pad_obs(self, state: GameState) -> np.ndarray:
        obs = state.to_observation()
        c, h, w = obs.shape
        padded = np.zeros((c, self.obs_size, self.obs_size), dtype=np.float32)
        ph, pw = min(h, self.obs_size), min(w, self.obs_size)
        padded[:, :ph, :pw] = obs[:, :ph, :pw]
        return padded.flatten()

    def load(self, path: str) -> None:
        from stable_baselines3 import PPO
        self.model = PPO.load(path)
        print(f"PPO model loaded from {path}")

    def reset(self) -> None:
        pass
