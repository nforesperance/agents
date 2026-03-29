"""Classical search solvers — A* and BFS."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Optional

import numpy as np

from config import FLOOR, WALL, TRAP, KEY, DOOR, GOAL, ENEMY, START, ACTION_DELTAS, UP, DOWN, LEFT, RIGHT, WAIT
from game.engine import GameState
from solvers.base import BaseSolver


# State for search: (player_row, player_col, frozenset_of_key_ids)
SearchState = tuple[int, int, frozenset]


class AStarSolver(BaseSolver):
    """A* solver with state = (position, keys_collected).

    Heuristic: Manhattan distance to nearest uncollected key (if any),
    plus Manhattan distance from that key to the goal. If all keys collected,
    Manhattan distance to goal (or door then goal).
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
                # distance to nearest uncollected key + key to goal
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

            # Goal check
            if grid[cr, cc] == GOAL or (cr, cc) == goal_pos:
                # Reconstruct path
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

                # Collect keys
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

        return []  # no solution found


class BFSSolver(BaseSolver):
    """BFS solver — guaranteed shortest path but explores everything."""

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
