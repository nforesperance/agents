"""Pygame visualization — side-by-side solver comparison view.

Supports both pre-computed solvers (A*, BFS, RL) that replay step-by-step,
and live solvers (LLM) that solve in real-time during visualization.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import pygame
import numpy as np

from config import (
    TILE_SIZE, COLORS, TILE_NAMES, KEY_COLORS,
    FLOOR, WALL, TRAP, KEY, DOOR, GOAL, ENEMY, START,
    ACTION_NAMES, FPS, SOLVER_STEP_DELAY,
    WINDOW_WIDTH, WINDOW_HEIGHT,
)
from game.engine import GameState


@dataclass
class SolverRun:
    """Tracks a solver's progress through a level."""
    name: str
    color: tuple[int, int, int]
    actions: list[int]
    states: list[GameState] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)
    current_step: int = 0
    total_reward: float = 0.0
    solve_time_ms: float = 0.0
    done: bool = False
    # Live solver support
    live_solver: object = None       # LLMSolver instance (None = precomputed)
    live_thinking: bool = False      # True while waiting for API response
    live_finished: bool = False      # True when live solver is done


class Visualizer:
    """Renders up to 3 solver runs side-by-side in Pygame."""

    def __init__(self, level: GameState, solver_runs: list[SolverRun]):
        pygame.init()
        pygame.display.set_caption("AI Puzzle Solver Showdown")

        self.level = level
        self.solver_runs = solver_runs
        self.num_solvers = len(solver_runs)

        # Precompute states for non-live solvers
        for run in self.solver_runs:
            if run.live_solver is None:
                self._precompute_states(run)
            else:
                # Initialize live solver with starting state
                state = self.level.copy()
                run.states = [state.copy()]
                run.step_rewards = [0.0]

        # Layout
        self.panel_width = WINDOW_WIDTH // max(self.num_solvers, 1)

        # Scale tiles to fit panel
        max_tile = min(
            (self.panel_width - 40) // level.cols,
            (WINDOW_HEIGHT - 160) // level.rows,
        )
        self.tile = max(12, min(max_tile, TILE_SIZE))

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 13)

        self.paused = False
        self.step_timer = 0
        self.all_done = False
        self.speed = SOLVER_STEP_DELAY

    def _precompute_states(self, run: SolverRun) -> None:
        """Simulate the actions and store each intermediate state."""
        state = self.level.copy()
        run.states = [state.copy()]
        run.step_rewards = [0.0]
        run.total_reward = 0.0
        for action in run.actions:
            if state.done:
                break
            state, reward, done = state.step(action)
            run.total_reward += reward
            run.states.append(state.copy())
            run.step_rewards.append(run.total_reward)
        run.done = state.done and state.won

    def _live_solver_thread(self, run: SolverRun) -> None:
        """Background thread that calls the LLM solver step by step."""
        from config import LLM_MAX_STEPS
        current_state = self.level.copy()
        run.total_reward = 0.0

        for step in range(LLM_MAX_STEPS):
            if current_state.done or self._shutdown.is_set():
                break

            run.live_thinking = True
            try:
                action = run.live_solver.solve_step(current_state)
            except Exception as e:
                print(f"  LLM error at step {step}: {e}")
                from config import WAIT
                action = WAIT
            run.live_thinking = False

            run.actions.append(action)
            current_state, reward, done = current_state.step(action)
            run.total_reward += reward
            run.states.append(current_state.copy())
            run.step_rewards.append(run.total_reward)
            run.current_step = len(run.states) - 1

            status = "WON!" if current_state.won else ("DEAD" if current_state.done else "")
            print(f"    Step {step+1}: {ACTION_NAMES[action]:5s} | HP: {current_state.health} | "
                  f"Keys: {len(current_state.keys_collected)}/{current_state.total_keys} {status}")

        run.live_finished = True
        run.done = current_state.won

    def run(self) -> None:
        """Main visualization loop."""
        import signal

        self._shutdown = threading.Event()

        def watchdog():
            self._shutdown.wait()
            time.sleep(2)
            os._exit(0)

        wd = threading.Thread(target=watchdog, daemon=True)
        wd.start()

        signal.signal(signal.SIGINT, lambda *_: self._shutdown.set())

        # Start live solver threads
        for run in self.solver_runs:
            if run.live_solver is not None:
                t = threading.Thread(target=self._live_solver_thread, args=(run,), daemon=True)
                t.start()

        running = True
        last_step_time = pygame.time.get_ticks()

        while running and not self._shutdown.is_set():
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT:
                        self._advance_all()
                    elif event.key == pygame.K_UP:
                        self.speed = max(50, self.speed - 50)
                    elif event.key == pygame.K_DOWN:
                        self.speed = min(1000, self.speed + 50)
                    elif event.key == pygame.K_r:
                        self._reset_all()

            # Auto-advance precomputed solvers
            now = pygame.time.get_ticks()
            if not self.paused and not self.all_done and now - last_step_time >= self.speed:
                self._advance_precomputed()
                last_step_time = now

            self._draw()
            pygame.display.flip()

        self._shutdown.set()
        pygame.quit()
        os._exit(0)

    def _advance_precomputed(self) -> None:
        """Advance only precomputed (non-live) solver runs."""
        all_finished = True
        for run in self.solver_runs:
            if run.live_solver is not None:
                # Live solvers advance themselves
                if not run.live_finished:
                    all_finished = False
                continue
            if run.current_step < len(run.states) - 1:
                run.current_step += 1
                all_finished = False
        self.all_done = all_finished

    def _advance_all(self) -> None:
        """Manual step advance (RIGHT key) — only advances precomputed."""
        self._advance_precomputed()

    def _reset_all(self) -> None:
        for run in self.solver_runs:
            if run.live_solver is None:
                run.current_step = 0
        self.all_done = False

    def _draw(self) -> None:
        self.screen.fill(COLORS["bg"])

        for i, run in enumerate(self.solver_runs):
            x_offset = i * self.panel_width
            self._draw_solver_panel(run, x_offset)

            # Separator line
            if i > 0:
                pygame.draw.line(
                    self.screen,
                    (60, 60, 80),
                    (x_offset, 0),
                    (x_offset, WINDOW_HEIGHT),
                    2,
                )

        # Bottom bar
        self._draw_bottom_bar()

    def _draw_solver_panel(self, run: SolverRun, x_offset: int) -> None:
        # For live solvers, current_step is updated by the thread
        step_idx = min(run.current_step, len(run.states) - 1)
        state = run.states[step_idx] if run.states else self.level

        # Title
        title = self.font_large.render(run.name, True, run.color)
        self.screen.blit(title, (x_offset + 10, 10))

        # Stats
        if run.live_solver is not None:
            total_steps = len(run.states) - 1 if run.states else 0
            step_text = f"Step: {total_steps}"
        else:
            total_steps = len(run.states) - 1 if run.states else 0
            step_text = f"Step: {step_idx}/{total_steps}"
        stats = self.font_small.render(step_text, True, COLORS["text"])
        self.screen.blit(stats, (x_offset + 10, 38))

        # Reward
        current_reward = run.step_rewards[step_idx] if step_idx < len(run.step_rewards) else run.total_reward
        reward_text = f"Reward: {current_reward:.0f}"
        rt = self.font_small.render(reward_text, True, COLORS["text"])
        self.screen.blit(rt, (x_offset + 10, 55))

        # Solve time / live status
        if run.live_solver is not None:
            if run.live_thinking:
                time_text = "THINKING..."
                tt = self.font_small.render(time_text, True, (255, 255, 100))
            elif run.live_finished:
                time_text = "DONE"
                tt = self.font_small.render(time_text, True, COLORS["text"])
            else:
                time_text = "WAITING..."
                tt = self.font_small.render(time_text, True, (150, 150, 170))
        else:
            if run.solve_time_ms < 1:
                time_text = f"Solve: {run.solve_time_ms * 1000:.0f}us"
            elif run.solve_time_ms < 100:
                time_text = f"Solve: {run.solve_time_ms:.1f}ms"
            else:
                time_text = f"Solve: {run.solve_time_ms:.0f}ms"
            tt = self.font_small.render(time_text, True, COLORS["text"])
        self.screen.blit(tt, (x_offset + 150, 55))

        # Status
        if run.live_solver is not None and run.live_thinking:
            status = "SOLVING..."
            status_color = (255, 255, 100)
        elif state.won:
            status = "WON!"
            status_color = (0, 255, 100)
        elif state.done:
            status = "FAILED"
            status_color = (255, 60, 60)
        elif run.live_solver is not None and not run.live_finished:
            status = "LIVE"
            status_color = (100, 255, 100)
        else:
            status = "RUNNING..."
            status_color = COLORS["text"]
        st = self.font.render(status, True, status_color)
        self.screen.blit(st, (x_offset + 150, 36))

        # HP bar with label
        max_hp = self.level.health
        hp_label = self.font_small.render(f"HP: {state.health}/{max_hp}", True, COLORS["text"])
        self.screen.blit(hp_label, (x_offset + 10, 73))
        hp_x = x_offset + 110
        hp_y = 76
        hp_w = self.panel_width - 130
        hp_h = 12
        pygame.draw.rect(self.screen, (60, 60, 60), (hp_x, hp_y, hp_w, hp_h))
        hp_ratio = max(0, state.health / max_hp) if max_hp > 0 else 0
        hp_color = (0, 200, 0) if hp_ratio > 0.5 else ((255, 165, 0) if hp_ratio > 0.25 else (255, 0, 0))
        pygame.draw.rect(self.screen, hp_color, (hp_x, hp_y, int(hp_w * hp_ratio), hp_h))
        pygame.draw.rect(self.screen, (80, 80, 100), (hp_x, hp_y, hp_w, hp_h), 1)

        # Grid
        grid_x = x_offset + (self.panel_width - state.cols * self.tile) // 2
        grid_y = 100

        for r in range(state.rows):
            for c in range(state.cols):
                rx = grid_x + c * self.tile
                ry = grid_y + r * self.tile
                tile = state.grid[r, c]

                # Base tile
                color = COLORS.get(tile, COLORS[FLOOR])
                pygame.draw.rect(self.screen, color, (rx, ry, self.tile - 1, self.tile - 1))

                # Tile decorations
                center = (rx + self.tile // 2, ry + self.tile // 2)
                small = self.tile // 4

                if tile == KEY:
                    pygame.draw.circle(self.screen, (255, 215, 0), center, small)
                elif tile == TRAP:
                    pygame.draw.line(self.screen, (255, 255, 255), (rx + 4, ry + 4), (rx + self.tile - 5, ry + self.tile - 5), 2)
                    pygame.draw.line(self.screen, (255, 255, 255), (rx + self.tile - 5, ry + 4), (rx + 4, ry + self.tile - 5), 2)
                elif tile == GOAL:
                    pygame.draw.polygon(self.screen, (255, 255, 255), [
                        (center[0], ry + 3),
                        (rx + self.tile - 4, ry + self.tile - 4),
                        (rx + 3, ry + self.tile - 4),
                    ])
                elif tile == DOOR:
                    pygame.draw.rect(self.screen, (200, 150, 50), (rx + 2, ry + 2, self.tile - 5, self.tile - 5), 2)

                # Grid lines
                pygame.draw.rect(self.screen, (50, 50, 70), (rx, ry, self.tile, self.tile), 1)

        # Enemies
        for er, ec in state.enemy_positions:
            ex = grid_x + ec * self.tile + self.tile // 2
            ey = grid_y + er * self.tile + self.tile // 2
            pygame.draw.circle(self.screen, COLORS[ENEMY], (ex, ey), self.tile // 3)

        # Player
        pr, pc = state.player_pos
        px = grid_x + pc * self.tile + self.tile // 2
        py = grid_y + pr * self.tile + self.tile // 2
        pygame.draw.circle(self.screen, run.color, (px, py), self.tile // 3)
        pygame.draw.circle(self.screen, (255, 255, 255), (px, py), self.tile // 3, 2)

        # Keys collected
        key_text = f"Keys: {len(state.keys_collected)}/{state.total_keys}"
        kt = self.font_small.render(key_text, True, (255, 215, 0))
        self.screen.blit(kt, (x_offset + 10, grid_y + state.rows * self.tile + 10))

    def _draw_bottom_bar(self) -> None:
        bar_y = WINDOW_HEIGHT - 35
        pygame.draw.rect(self.screen, COLORS["panel"], (0, bar_y, WINDOW_WIDTH, 35))

        controls = "SPACE=pause  RIGHT=step  UP/DOWN=speed  R=restart  Q=quit"
        ct = self.font_small.render(controls, True, (150, 150, 170))
        self.screen.blit(ct, (10, bar_y + 8))

        speed_text = f"Speed: {self.speed}ms"
        spt = self.font_small.render(speed_text, True, COLORS["text"])
        self.screen.blit(spt, (WINDOW_WIDTH - 140, bar_y + 8))

        if self.paused:
            pt = self.font.render("PAUSED", True, (255, 255, 100))
            self.screen.blit(pt, (WINDOW_WIDTH // 2 - 40, bar_y + 6))
