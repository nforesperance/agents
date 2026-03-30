"""Procedural level generator — creates solvable puzzle levels."""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from config import (
    FLOOR, WALL, TRAP, KEY, DOOR, GOAL, START, ENEMY,
    GRID_MIN, GRID_MAX, DEFAULT_GRID_SIZE, UP, DOWN, LEFT, RIGHT,
)
from game.engine import GameState


class LevelGenerator:
    """Generates random solvable grid-based puzzle levels."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        size: int = DEFAULT_GRID_SIZE,
        num_keys: int = 1,
        num_traps: int = 3,
        num_enemies: int = 1,
        difficulty: int = 1,
    ) -> GameState:
        """Create a random level guaranteed to be solvable.

        Difficulty 1-5 scales grid size, keys, traps, enemies.
        """
        if difficulty > 1:
            size = max(size, size + (difficulty - 1))
            num_keys = min(4, num_keys + difficulty - 1)
            num_traps = num_traps + difficulty * 2
            num_enemies = min(4, num_enemies + difficulty - 1)

        # Scale hazards proportionally to grid area
        area_ratio = (size * size) / (DEFAULT_GRID_SIZE * DEFAULT_GRID_SIZE)
        num_traps = max(num_traps, int(num_traps * area_ratio))
        num_enemies = max(num_enemies, int(num_enemies * area_ratio))

        grid = self._generate_maze(size, size)

        # Collect floor cells
        floor_cells = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if grid[r, c] == FLOOR
        ]
        self.rng.shuffle(floor_cells)

        # Place start
        start_pos = floor_cells.pop()
        grid[start_pos] = START

        # Place goal (try to pick a far cell)
        floor_cells.sort(key=lambda p: abs(p[0] - start_pos[0]) + abs(p[1] - start_pos[1]))
        goal_pos = floor_cells.pop()  # farthest
        grid[goal_pos] = GOAL

        # Re-shuffle so remaining elements spread evenly
        self.rng.shuffle(floor_cells)

        # Place keys
        actual_keys = min(num_keys, len(floor_cells))
        key_positions = []
        for _ in range(actual_keys):
            if floor_cells:
                kp = floor_cells.pop()
                grid[kp] = KEY
                key_positions.append(kp)

        # Place door (on the path to goal if possible)
        door_placed = False
        if actual_keys > 0 and floor_cells:
            # try to place door between start and goal
            mid_r = (start_pos[0] + goal_pos[0]) // 2
            mid_c = (start_pos[1] + goal_pos[1]) // 2
            floor_cells.sort(key=lambda p: abs(p[0] - mid_r) + abs(p[1] - mid_c))
            for i, cell in enumerate(floor_cells):
                # check it doesn't block all paths when locked
                grid[cell] = DOOR
                if self._is_reachable(grid, start_pos, goal_pos, has_keys=True):
                    floor_cells.pop(i)
                    door_placed = True
                    break
                grid[cell] = FLOOR

        # Re-shuffle before placing hazards to spread them evenly
        self.rng.shuffle(floor_cells)

        # Place traps
        actual_traps = min(num_traps, len(floor_cells))
        for _ in range(actual_traps):
            if floor_cells:
                tp = floor_cells.pop()
                grid[tp] = TRAP

        # Place enemies
        enemy_positions = []
        enemy_directions = []
        actual_enemies = min(num_enemies, len(floor_cells))
        for _ in range(actual_enemies):
            if floor_cells:
                ep = floor_cells.pop()
                enemy_positions.append(ep)
                enemy_directions.append(self.rng.choice([UP, DOWN, LEFT, RIGHT]))

        # Verify solvability
        if not self._is_reachable(grid, start_pos, goal_pos, has_keys=True):
            # Fallback: regenerate (rare with maze-based generation)
            return self.generate(size, num_keys, num_traps, num_enemies, difficulty)

        # Scale HP based on grid size and hazard count
        # Base: 100 HP for a 9x9 grid. Scale with area and hazard density.
        base_hp = 100
        hp = base_hp + actual_traps * 15 + actual_enemies * 20 + max(0, size - 9) * 5

        return GameState(
            grid=grid,
            player_pos=start_pos,
            health=hp,
            total_keys=actual_keys,
            enemy_positions=enemy_positions,
            enemy_directions=enemy_directions,
        )

    def _generate_maze(self, rows: int, cols: int) -> np.ndarray:
        """Generate a maze using randomized Prim's algorithm.

        Returns a grid where carved paths are FLOOR and walls are WALL.
        """
        grid = np.full((rows, cols), WALL, dtype=np.int32)

        # Start carving from (1,1)
        sr, sc = 1, 1
        grid[sr, sc] = FLOOR
        frontiers: list[tuple[int, int]] = []

        def add_frontiers(r: int, c: int) -> None:
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 0 < nr < rows - 1 and 0 < nc < cols - 1 and grid[nr, nc] == WALL:
                    frontiers.append((nr, nc))

        add_frontiers(sr, sc)

        while frontiers:
            idx = self.rng.randrange(len(frontiers))
            fr, fc = frontiers.pop(idx)

            if grid[fr, fc] != FLOOR:
                # Find carved neighbors
                neighbors = []
                for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    nr, nc = fr + dr, fc + dc
                    if 0 < nr < rows - 1 and 0 < nc < cols - 1 and grid[nr, nc] == FLOOR:
                        neighbors.append((nr, nc, (fr + nr) // 2, (fc + nc) // 2))

                if neighbors:
                    nr, nc, wr, wc = self.rng.choice(neighbors)
                    grid[fr, fc] = FLOOR
                    grid[wr, wc] = FLOOR
                    add_frontiers(fr, fc)

        # Open up some extra passages for more interesting puzzles
        extra = (rows * cols) // 8
        for _ in range(extra):
            r = self.rng.randint(1, rows - 2)
            c = self.rng.randint(1, cols - 2)
            if grid[r, c] == WALL:
                # only open if it won't create large open areas
                adj_floor = sum(
                    1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= r + dr < rows and 0 <= c + dc < cols and grid[r + dr, c + dc] == FLOOR
                )
                if 1 <= adj_floor <= 2:
                    grid[r, c] = FLOOR

        return grid

    @staticmethod
    def _is_reachable(
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        has_keys: bool = False,
    ) -> bool:
        """BFS to check if goal is reachable from start."""
        rows, cols = grid.shape
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr, nc] != WALL
                    and (grid[nr, nc] != DOOR or has_keys)
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def generate_batch(self, count: int, **kwargs) -> list[GameState]:
        """Generate multiple levels for training/benchmarking."""
        return [self.generate(**kwargs) for _ in range(count)]
