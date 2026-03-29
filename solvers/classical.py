"""Classical search solvers — A*, BFS, and cost-aware A*.

This module implements three classical search strategies:
- **BFS**: Breadth-first search. Finds the shortest path (fewest steps)
  but ignores all hazard costs (traps, enemies).
- **A***: A-star search with Manhattan distance heuristic. Finds the
  shortest path using an admissible heuristic, but also ignores hazards.
- **A* Safe**: Cost-aware A* that penalizes traps and enemy-adjacent
  cells in the path cost, finding safer (though possibly longer) routes.

All solvers operate on the static grid and treat the search state as
(player_row, player_col, frozenset_of_collected_key_ids).
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Optional

import numpy as np

from config import (
    FLOOR, WALL, TRAP, KEY, DOOR, GOAL, ENEMY, START,
    ACTION_DELTAS, UP, DOWN, LEFT, RIGHT, WAIT,
)
from game.engine import GameState
from solvers.base import BaseSolver


# State for search: (player_row, player_col, frozenset_of_key_ids)
SearchState = tuple[int, int, frozenset]


class AStarSolver(BaseSolver):
    """A* solver with state = (position, keys_collected).

    Uses Manhattan distance as the heuristic. Finds the shortest path
    but does NOT account for traps or enemies — it treats them as
    free tiles. This makes it fast but potentially fatal on hazardous maps.

    Heuristic: Manhattan distance to nearest uncollected key (if any),
    plus Manhattan distance from that key to the goal. If all keys are
    collected, straight Manhattan distance to the goal.
    """

    name = "A*"
    color = "classical"

    def solve(self, state: GameState) -> list[int]:
        grid = state.grid
        rows, cols = grid.shape
        start = state.player_pos
        total_keys = state.total_keys

        # Precompute key and goal positions
        key_positions = []
        goal_pos = None
        door_pos = None
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == KEY:
                    key_positions.append((r, c))
                elif grid[r, c] == GOAL:
                    goal_pos = (r, c)
                elif grid[r, c] == DOOR:
                    door_pos = (r, c)

        if goal_pos is None:
            return []

        initial_keys = frozenset(state.keys_collected)
        initial_state: SearchState = (start[0], start[1], initial_keys)

        def heuristic(s: SearchState) -> float:
            r, c, keys = s
            if len(keys) < total_keys:
                uncollected = [
                    kp for i, kp in enumerate(key_positions) if i not in keys
                ]
                if uncollected:
                    nearest = min(uncollected, key=lambda kp: abs(kp[0] - r) + abs(kp[1] - c))
                    return (
                        abs(nearest[0] - r) + abs(nearest[1] - c)
                        + abs(goal_pos[0] - nearest[0]) + abs(goal_pos[1] - nearest[1])
                    )
            return abs(goal_pos[0] - r) + abs(goal_pos[1] - c)

        # A* search
        g_score = {initial_state: 0}
        f_score = {initial_state: heuristic(initial_state)}
        open_set: list[tuple[float, int, SearchState]] = [(f_score[initial_state], 0, initial_state)]
        came_from: dict[SearchState, tuple[SearchState, int]] = {}
        counter = 1

        while open_set:
            _, _, current = heapq.heappop(open_set)
            cr, cc, ckeys = current

            if grid[cr, cc] == GOAL or (cr, cc) == goal_pos:
                actions = []
                node = current
                while node in came_from:
                    prev, action = came_from[node]
                    actions.append(action)
                    node = prev
                actions.reverse()
                return actions

            for action in [UP, DOWN, LEFT, RIGHT]:
                dr, dc = ACTION_DELTAS[action]
                nr, nc = cr + dr, cc + dc

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                tile = grid[nr, nc]
                if tile == WALL:
                    continue
                if tile == DOOR and len(ckeys) < total_keys:
                    continue

                new_keys = ckeys
                if tile == KEY:
                    for i, kp in enumerate(key_positions):
                        if kp == (nr, nc) and i not in ckeys:
                            new_keys = ckeys | {i}
                            break

                neighbor: SearchState = (nr, nc, new_keys)
                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = (current, action)
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        return []


class AStarSafeSolver(BaseSolver):
    """Cost-aware A* solver that avoids traps and enemies.

    Unlike the basic A*, this solver assigns higher movement costs to
    hazardous tiles:
    - Trap tiles cost 10 extra (matching the game's reward penalty).
    - Tiles adjacent to an enemy position cost 5 extra.
    - Normal floor tiles cost 1.

    This produces safer paths that may be longer in steps but avoid
    taking damage, resulting in higher survival rates on hard maps.

    The heuristic remains admissible (Manhattan distance) so the
    solution is still optimal with respect to the weighted cost.
    """

    name = "A* Safe"
    color = "classical"

    def solve(self, state: GameState) -> list[int]:
        grid = state.grid
        rows, cols = grid.shape
        start = state.player_pos
        total_keys = state.total_keys

        key_positions = []
        goal_pos = None
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == KEY:
                    key_positions.append((r, c))
                elif grid[r, c] == GOAL:
                    goal_pos = (r, c)

        if goal_pos is None:
            return []

        # Precompute enemy danger zones (enemy cells + adjacent cells)
        enemy_cells = set()
        for er, ec in state.enemy_positions:
            enemy_cells.add((er, ec))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                enemy_cells.add((er + dr, ec + dc))

        def tile_cost(r: int, c: int) -> float:
            """Movement cost for stepping onto tile (r, c)."""
            tile = grid[r, c]
            cost = 1.0
            if tile == TRAP:
                cost += 10.0  # strongly discourage traps
            if (r, c) in enemy_cells:
                cost += 5.0   # discourage enemy-adjacent tiles
            return cost

        def heuristic(s: SearchState) -> float:
            r, c, keys = s
            if len(keys) < total_keys:
                uncollected = [
                    kp for i, kp in enumerate(key_positions) if i not in keys
                ]
                if uncollected:
                    nearest = min(uncollected, key=lambda kp: abs(kp[0] - r) + abs(kp[1] - c))
                    return (
                        abs(nearest[0] - r) + abs(nearest[1] - c)
                        + abs(goal_pos[0] - nearest[0]) + abs(goal_pos[1] - nearest[1])
                    )
            return abs(goal_pos[0] - r) + abs(goal_pos[1] - c)

        initial_keys = frozenset(state.keys_collected)
        initial_state: SearchState = (start[0], start[1], initial_keys)

        g_score = {initial_state: 0.0}
        open_set: list[tuple[float, int, SearchState]] = [
            (heuristic(initial_state), 0, initial_state)
        ]
        came_from: dict[SearchState, tuple[SearchState, int]] = {}
        counter = 1

        while open_set:
            _, _, current = heapq.heappop(open_set)
            cr, cc, ckeys = current

            if (cr, cc) == goal_pos:
                actions = []
                node = current
                while node in came_from:
                    prev, action = came_from[node]
                    actions.append(action)
                    node = prev
                actions.reverse()
                return actions

            for action in [UP, DOWN, LEFT, RIGHT]:
                dr, dc = ACTION_DELTAS[action]
                nr, nc = cr + dr, cc + dc

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                tile = grid[nr, nc]
                if tile == WALL:
                    continue
                if tile == DOOR and len(ckeys) < total_keys:
                    continue

                new_keys = ckeys
                if tile == KEY:
                    for i, kp in enumerate(key_positions):
                        if kp == (nr, nc) and i not in ckeys:
                            new_keys = ckeys | {i}
                            break

                neighbor: SearchState = (nr, nc, new_keys)
                tentative_g = g_score[current] + tile_cost(nr, nc)

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = (current, action)
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        return []


class BFSSolver(BaseSolver):
    """Breadth-first search solver — guaranteed shortest path.

    BFS explores all states level by level, guaranteeing the fewest
    number of steps. However, it does NOT consider hazard costs at all,
    so it may route through traps and enemies.

    This serves as a baseline to compare against smarter solvers.
    """

    name = "BFS"
    color = "classical"

    def solve(self, state: GameState) -> list[int]:
        grid = state.grid
        rows, cols = grid.shape
        start = state.player_pos
        total_keys = state.total_keys

        key_positions = []
        goal_pos = None
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == KEY:
                    key_positions.append((r, c))
                elif grid[r, c] == GOAL:
                    goal_pos = (r, c)

        if goal_pos is None:
            return []

        initial_keys = frozenset(state.keys_collected)
        initial_state: SearchState = (start[0], start[1], initial_keys)

        visited: set[SearchState] = {initial_state}
        queue: deque[tuple[SearchState, list[int]]] = deque([(initial_state, [])])

        while queue:
            (cr, cc, ckeys), actions = queue.popleft()

            for action in [UP, DOWN, LEFT, RIGHT]:
                dr, dc = ACTION_DELTAS[action]
                nr, nc = cr + dr, cc + dc

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                tile = grid[nr, nc]
                if tile == WALL:
                    continue
                if tile == DOOR and len(ckeys) < total_keys:
                    continue

                new_keys = ckeys
                if tile == KEY:
                    for i, kp in enumerate(key_positions):
                        if kp == (nr, nc) and i not in ckeys:
                            new_keys = ckeys | {i}
                            break

                neighbor: SearchState = (nr, nc, new_keys)
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                new_actions = actions + [action]

                if (nr, nc) == goal_pos:
                    return new_actions

                queue.append((neighbor, new_actions))

        return []
