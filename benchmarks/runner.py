"""Benchmark runner — evaluates solvers across multiple levels."""

from __future__ import annotations

import time
from dataclasses import dataclass

from game.engine import GameState
from game.level_generator import LevelGenerator
from solvers.base import BaseSolver
from ui.dashboard import BenchmarkResult, plot_benchmark


def run_solver_on_level(solver: BaseSolver, level: GameState) -> dict:
    """Run a single solver on a single level, return metrics."""
    start = time.perf_counter()
    actions = solver.solve(level.copy())
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Simulate the actions to get final state
    state = level.copy()
    total_reward = 0.0
    steps = 0
    for action in actions:
        if state.done:
            break
        state, reward, done = state.step(action)
        total_reward += reward
        steps += 1

    return {
        "solved": state.won,
        "steps": steps,
        "reward": total_reward,
        "time_ms": elapsed_ms,
        "health": state.health,
    }


def benchmark(
    solvers: list[BaseSolver],
    num_levels: int = 20,
    grid_size: int = 9,
    difficulty: int = 1,
    seed: int = 42,
    show_dashboard: bool = True,
    save_path: str | None = None,
) -> list[BenchmarkResult]:
    """Run all solvers across the same set of generated levels."""
    gen = LevelGenerator(seed=seed)
    levels = gen.generate_batch(num_levels, size=grid_size, difficulty=difficulty)

    results: list[BenchmarkResult] = []

    for solver in solvers:
        print(f"\nBenchmarking {solver.name}...")
        steps_list = []
        rewards_list = []
        times_list = []
        solved = 0

        for i, level in enumerate(levels):
            solver.reset()
            metrics = run_solver_on_level(solver, level)

            steps_list.append(metrics["steps"])
            rewards_list.append(metrics["reward"])
            times_list.append(metrics["time_ms"])
            if metrics["solved"]:
                solved += 1

            status = "SOLVED" if metrics["solved"] else "FAILED"
            print(f"  Level {i+1:2d}: {status} | {metrics['steps']:3d} steps | "
                  f"{metrics['reward']:7.1f} reward | {metrics['time_ms']:7.1f}ms")

        result = BenchmarkResult(
            solver_name=solver.name,
            levels_solved=solved,
            levels_total=num_levels,
            avg_steps=sum(steps_list) / len(steps_list) if steps_list else 0,
            avg_reward=sum(rewards_list) / len(rewards_list) if rewards_list else 0,
            avg_time_ms=sum(times_list) / len(times_list) if times_list else 0,
            solve_rate=solved / num_levels if num_levels > 0 else 0,
            steps_list=steps_list,
            rewards_list=rewards_list,
            times_list=times_list,
        )
        results.append(result)
        print(f"  => {solver.name}: {solved}/{num_levels} solved "
              f"({result.solve_rate*100:.1f}%), avg {result.avg_steps:.1f} steps, "
              f"avg {result.avg_time_ms:.1f}ms")

    if show_dashboard:
        plot_benchmark(results, save_path=save_path)

    return results
