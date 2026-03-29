"""Benchmark dashboard — matplotlib-based results visualization."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("TkAgg")


@dataclass
class BenchmarkResult:
    solver_name: str
    levels_solved: int
    levels_total: int
    avg_steps: float
    avg_reward: float
    avg_time_ms: float
    solve_rate: float  # 0-1
    steps_list: list[int]
    rewards_list: list[float]
    times_list: list[float]


def plot_benchmark(results: list[BenchmarkResult], save_path: str | None = None) -> None:
    """Create a comprehensive benchmark comparison dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("AI Puzzle Solver Showdown — Benchmark Results", fontsize=16, fontweight="bold")

    colors = {"A*": "#00c8ff", "BFS": "#0088cc", "RL (DQN)": "#ff6432", "LLM": "#64ff64"}

    names = [r.solver_name for r in results]
    solver_colors = [colors.get(n, "#aaaaaa") for n in names]

    # 1. Solve rate bar chart
    ax = axes[0, 0]
    rates = [r.solve_rate * 100 for r in results]
    bars = ax.bar(names, rates, color=solver_colors)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Success Rate")
    ax.set_ylim(0, 110)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{rate:.1f}%", ha="center", fontsize=10)

    # 2. Average steps
    ax = axes[0, 1]
    avg_steps = [r.avg_steps for r in results]
    bars = ax.bar(names, avg_steps, color=solver_colors)
    ax.set_ylabel("Avg Steps")
    ax.set_title("Efficiency (fewer = better)")
    for bar, s in zip(bars, avg_steps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{s:.1f}", ha="center", fontsize=10)

    # 3. Average time
    ax = axes[0, 2]
    avg_times = [r.avg_time_ms for r in results]
    bars = ax.bar(names, avg_times, color=solver_colors)
    ax.set_ylabel("Avg Time (ms)")
    ax.set_title("Computation Speed")
    for bar, t in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.0f}", ha="center", fontsize=10)

    # 4. Reward distribution (box plot)
    ax = axes[1, 0]
    reward_data = [r.rewards_list for r in results]
    bp = ax.boxplot(reward_data, labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], solver_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution")

    # 5. Steps distribution
    ax = axes[1, 1]
    steps_data = [r.steps_list for r in results]
    bp = ax.boxplot(steps_data, labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], solver_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Steps")
    ax.set_title("Steps Distribution")

    # 6. Radar/summary chart
    ax = axes[1, 2]
    categories = ["Solve Rate", "Efficiency", "Speed", "Avg Reward"]
    for i, r in enumerate(results):
        # Normalize each metric to 0-1
        max_steps = max(rr.avg_steps for rr in results) or 1
        max_time = max(rr.avg_time_ms for rr in results) or 1
        max_reward = max(abs(rr.avg_reward) for rr in results) or 1

        values = [
            r.solve_rate,
            1 - (r.avg_steps / max_steps) if max_steps > 0 else 0,
            1 - (r.avg_time_ms / max_time) if max_time > 0 else 0,
            (r.avg_reward + max_reward) / (2 * max_reward) if max_reward > 0 else 0.5,
        ]
        x = np.arange(len(categories))
        ax.bar(x + i * 0.2 - 0.3, values, width=0.18, label=r.solver_name,
               color=solver_colors[i], alpha=0.8)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_title("Overall Comparison (normalized)")
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")

    plt.show()
