"""Core game engine — handles state, actions, and rules."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import (
    FLOOR, WALL, TRAP, KEY, DOOR, GOAL, ENEMY, START,
    ACTION_DELTAS, UP, DOWN, LEFT, RIGHT, WAIT,
    TILE_NAMES, RL_MAX_STEPS,
)


@dataclass
class GameState:
    """Immutable-ish snapshot of the puzzle world."""

    grid: np.ndarray                          # 2-D int array of tile types
    player_pos: tuple[int, int]               # (row, col)
    keys_collected: list[int] = field(default_factory=list)
    health: int = 100
    steps: int = 0
    max_steps: int = RL_MAX_STEPS
    done: bool = False
    won: bool = False
    total_keys: int = 0                       # how many keys exist in level
    enemy_positions: list[tuple[int, int]] = field(default_factory=list)
    enemy_directions: list[int] = field(default_factory=list)  # current move dir

    # --- helpers --------------------------------------------------------

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    def copy(self) -> GameState:
        return GameState(
            grid=self.grid.copy(),
            player_pos=self.player_pos,
            keys_collected=list(self.keys_collected),
            health=self.health,
            steps=self.steps,
            max_steps=self.max_steps,
            done=self.done,
            won=self.won,
            total_keys=self.total_keys,
            enemy_positions=list(self.enemy_positions),
            enemy_directions=list(self.enemy_directions),
        )

    # --- core API -------------------------------------------------------

    def step(self, action: int) -> tuple[GameState, float, bool]:
        """Apply *action*, return (new_state, reward, done).

        Reward scheme (dense, to help RL learn):
          +100  reach goal
           +10  pick up a key
            -1  each step (encourages efficiency)
           -10  hit a trap
           -20  hit an enemy
           -50  timeout
        """
        ns = self.copy()
        ns.steps += 1

        dr, dc = ACTION_DELTAS[action]
        nr, nc = ns.player_pos[0] + dr, ns.player_pos[1] + dc

        reward = -1.0  # step cost

        # --- boundary / wall check ---
        if not (0 <= nr < ns.rows and 0 <= nc < ns.cols) or ns.grid[nr, nc] == WALL:
            # bump into wall — stay in place
            ns._move_enemies()
            reward += ns._check_enemy_collision()
            if ns.steps >= ns.max_steps:
                ns.done = True
                reward -= 50
            return ns, reward, ns.done

        tile = ns.grid[nr, nc]

        # --- door check (need all keys) ---
        if tile == DOOR:
            if len(ns.keys_collected) < ns.total_keys:
                # door locked — stay
                ns._move_enemies()
                reward += ns._check_enemy_collision()
                if ns.steps >= ns.max_steps:
                    ns.done = True
                    reward -= 50
                return ns, reward, ns.done
            else:
                ns.grid[nr, nc] = FLOOR  # open the door

        # --- move player ---
        ns.player_pos = (nr, nc)

        # --- check if player walked into an enemy ---
        reward += ns._check_enemy_collision()
        if ns.done:
            return ns, reward, True

        # --- tile effects ---
        if tile == KEY:
            key_id = len(ns.keys_collected)
            ns.keys_collected.append(key_id)
            ns.grid[nr, nc] = FLOOR
            reward += 10

        elif tile == TRAP:
            ns.health -= 25
            reward -= 10
            ns.grid[nr, nc] = FLOOR  # trap triggers once
            if ns.health <= 0:
                ns.done = True
                return ns, reward, True

        elif tile == GOAL:
            ns.done = True
            ns.won = True
            reward += 100
            return ns, reward, True

        # --- enemies move, then check collision again ---
        ns._move_enemies()
        reward += ns._check_enemy_collision()

        # --- timeout ---
        if ns.steps >= ns.max_steps:
            ns.done = True
            reward -= 50

        return ns, reward, ns.done

    # --- enemy logic (simple patrol) ------------------------------------

    def _move_enemies(self) -> None:
        directions = [UP, DOWN, LEFT, RIGHT]
        for i, (er, ec) in enumerate(self.enemy_positions):
            d = self.enemy_directions[i]
            dr, dc = ACTION_DELTAS[d]
            nr, nc = er + dr, ec + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == FLOOR:
                self.enemy_positions[i] = (nr, nc)
            else:
                # reverse direction
                opposites = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
                self.enemy_directions[i] = opposites[d]

    def _check_enemy_collision(self) -> float:
        for ep in self.enemy_positions:
            if ep == self.player_pos:
                self.health -= 30
                if self.health <= 0:
                    self.done = True
                return -20.0
        return 0.0

    # --- observation for RL (flat numpy) --------------------------------

    def to_observation(self) -> np.ndarray:
        """Return a multi-channel grid observation for the RL agent.

        Channels:
          0 — tile type (normalized)
          1 — player position (one-hot)
          2 — enemy positions (one-hot)
          3 — keys collected (scalar broadcast)
          4 — health (scalar broadcast)
        """
        h, w = self.rows, self.cols
        obs = np.zeros((5, h, w), dtype=np.float32)

        # channel 0: tiles normalized to [0, 1]
        obs[0] = self.grid.astype(np.float32) / 7.0

        # channel 1: player
        obs[1, self.player_pos[0], self.player_pos[1]] = 1.0

        # channel 2: enemies
        for er, ec in self.enemy_positions:
            obs[2, er, ec] = 1.0

        # channel 3: keys ratio
        if self.total_keys > 0:
            obs[3] = len(self.keys_collected) / self.total_keys

        # channel 4: health
        obs[4] = self.health / 100.0

        return obs

    # --- text representation (for LLM) ---------------------------------

    def to_text(self) -> str:
        """Human-readable grid for the LLM solver."""
        symbols = {
            FLOOR: ".",
            WALL: "#",
            TRAP: "X",
            KEY: "K",
            DOOR: "D",
            GOAL: "G",
            START: "S",
        }
        lines = []
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                if (r, c) == self.player_pos:
                    row += "P"
                elif (r, c) in self.enemy_positions:
                    row += "E"
                else:
                    row += symbols.get(self.grid[r, c], "?")
            lines.append(row)

        header = (
            f"Grid {self.rows}x{self.cols} | "
            f"Keys: {len(self.keys_collected)}/{self.total_keys} | "
            f"HP: {self.health} | Steps: {self.steps}/{self.max_steps}"
        )
        legend = "Legend: P=player #=wall .=floor X=trap K=key D=door G=goal E=enemy"
        return f"{header}\n{legend}\n" + "\n".join(lines)
