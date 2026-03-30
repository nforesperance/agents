#!/usr/bin/env python3
"""
AI Puzzle Solver Showdown
=========================
Compare Classical AI (A*) vs RL (DQN) vs LLM (Claude/ChatGPT)
on procedurally generated puzzle levels.

Usage:
    python main.py demo                        # Visual demo with all solvers
    python main.py demo --solvers astar rl     # Specific solvers only
    python main.py benchmark                   # Run full benchmark
    python main.py train                       # Train RL agent
    python main.py play                        # Play manually
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import sys
import time

from config import DEFAULT_GRID_SIZE, COLORS
from game.engine import GameState
from game.level_generator import LevelGenerator


def cmd_demo(args) -> None:
    """Visual demo — watch solvers compete side-by-side."""
    from solvers.classical import AStarSolver, AStarSafeSolver, BFSSolver
    from ui.visualizer import Visualizer, SolverRun

    import random
    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    gen = LevelGenerator(seed=seed)
    if args.simple:
        level = gen.generate(size=args.grid_size, num_keys=0, num_traps=0, num_enemies=0, difficulty=1)
    else:
        level = gen.generate(size=args.grid_size, difficulty=args.difficulty)

    mode = "simple (goal only)" if args.simple else f"difficulty {args.difficulty}"
    print(f"Seed: {seed}  |  Grid: {args.grid_size}x{args.grid_size}  |  {mode}")
    print(f"Replay: python main.py demo --seed {seed} --grid-size {args.grid_size} --difficulty {args.difficulty}{'--simple' if args.simple else ''}")
    print()
    print(level.to_text())
    print()

    solver_runs: list[SolverRun] = []

    solver_map = {
        "astar": ("A*", AStarSolver, COLORS["classical"]),
        "astar-safe": ("A* Safe", AStarSafeSolver, (0, 150, 200)),
        "bfs": ("BFS", BFSSolver, COLORS["classical"]),
    }

    # Classical & BFS solvers
    for key in args.solvers:
        if key in solver_map:
            name, cls, color = solver_map[key]
            solver = cls()
            print(f"Solving with {name}...")
            start = time.perf_counter()
            actions = solver.solve(level)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  {name}: {len(actions)} actions in {elapsed:.1f}ms")
            solver_runs.append(SolverRun(
                name=name, color=color, actions=actions, solve_time_ms=elapsed,
            ))

    # RL solver
    if "rl" in args.solvers:
        from solvers.rl_solver import RLSolver
        if args.rl_snapshot and os.path.exists(args.rl_snapshot):
            model_path = args.rl_snapshot
        else:
            model_path = os.path.join("models", f"dqn_grid{args.grid_size}_d{args.difficulty}.pt")

        if os.path.exists(model_path):
            rl = RLSolver(grid_size=args.grid_size, model_path=model_path)
        else:
            print(f"  No trained model at {model_path} — using untrained agent")
            rl = RLSolver(grid_size=args.grid_size)

        print(f"Solving with RL (DQN)...")
        start = time.perf_counter()
        actions = rl.solve(level)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  RL: {len(actions)} actions in {elapsed:.1f}ms")
        solver_runs.append(SolverRun(
            name="RL (DQN)", color=COLORS["rl"], actions=actions, solve_time_ms=elapsed,
        ))

    # LLM solver (runs live during visualization)
    if "llm" in args.solvers:
        from solvers.llm_solver import LLMSolver
        try:
            llm = LLMSolver(provider=args.llm_provider, model=args.llm_model)
            print(f"LLM solver ({args.llm_provider}: {llm.model}) will solve LIVE in the visualizer")
            solver_runs.append(SolverRun(
                name=f"LLM ({args.llm_provider})", color=COLORS["llm"],
                actions=[], live_solver=llm,
            ))
        except Exception as e:
            print(f"  LLM solver error: {e}")
            print(f"  Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY env var")

    if not solver_runs:
        print("No solvers selected! Use --solvers astar rl llm")
        return

    print(f"\nLaunching visualization with {len(solver_runs)} solver(s)...")
    viz = Visualizer(level, solver_runs)
    viz.run()


def cmd_benchmark(args) -> None:
    """Run full benchmark across multiple levels."""
    from solvers.classical import AStarSolver, AStarSafeSolver, BFSSolver
    from benchmarks.runner import benchmark

    solvers = []

    if "astar" in args.solvers:
        solvers.append(AStarSolver())
    if "astar-safe" in args.solvers:
        solvers.append(AStarSafeSolver())
    if "bfs" in args.solvers:
        solvers.append(BFSSolver())
    if "rl" in args.solvers:
        from solvers.rl_solver import RLSolver
        if args.rl_snapshot and os.path.exists(args.rl_snapshot):
            model_path = args.rl_snapshot
        else:
            model_path = os.path.join("models", f"dqn_grid{args.grid_size}_d{args.difficulty}.pt")
        solvers.append(RLSolver(grid_size=args.grid_size, model_path=model_path if os.path.exists(model_path) else None))
    if "llm" in args.solvers:
        from solvers.llm_solver import LLMSolver
        try:
            solvers.append(LLMSolver(provider=args.llm_provider, model=args.llm_model))
        except Exception as e:
            print(f"LLM solver not available: {e}")

    if not solvers:
        print("No solvers to benchmark!")
        return

    benchmark(
        solvers=solvers,
        num_levels=args.num_levels,
        grid_size=args.grid_size,
        difficulty=args.difficulty,
        seed=args.seed,
        save_path=args.save_plot,
    )


def cmd_train(args) -> None:
    """Train the RL agent."""
    from training.train_rl import train
    train(
        episodes=args.episodes,
        grid_size=args.grid_size,
        difficulty=args.difficulty,
        save_dir="models",
        save_every=args.save_every,
        snapshot_every=args.snapshot_every,
        curriculum=not args.no_curriculum,
    )


def cmd_play(args) -> None:
    """Play the puzzle manually with keyboard."""
    import pygame
    from config import TILE_SIZE, COLORS, UP, DOWN, LEFT, RIGHT, WAIT, FLOOR, WALL, TRAP, KEY, DOOR, GOAL, ENEMY, START

    import random
    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    gen = LevelGenerator(seed=seed)
    level = gen.generate(size=args.grid_size, difficulty=args.difficulty)

    print(f"Seed: {seed}  |  Grid: {args.grid_size}x{args.grid_size}  |  Difficulty: {args.difficulty}")
    print(f"Replay: python main.py play --seed {seed} --grid-size {args.grid_size} --difficulty {args.difficulty}")
    print()

    pygame.init()
    tile = 50
    width = level.cols * tile + 300
    height = level.rows * tile + 100
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("AI Puzzle — Manual Play")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    import signal
    import threading

    shutdown = threading.Event()

    def watchdog():
        shutdown.wait()
        time.sleep(2)
        os._exit(0)

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()

    signal.signal(signal.SIGINT, lambda *_: shutdown.set())

    state = level.copy()
    total_reward = 0.0

    running = True
    while running and not shutdown.is_set():
            clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    action = None
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        action = UP
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        action = DOWN
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        action = LEFT
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        action = RIGHT
                    elif event.key == pygame.K_SPACE:
                        action = WAIT
                    elif event.key == pygame.K_r:
                        state = level.copy()
                        total_reward = 0.0

                    if action is not None and not state.done:
                        state, reward, done = state.step(action)
                        total_reward += reward

            # Draw
            screen.fill(COLORS["bg"])

            # Grid
            gx, gy = 20, 60
            for r in range(state.rows):
                for c in range(state.cols):
                    rx = gx + c * tile
                    ry = gy + r * tile
                    t = state.grid[r, c]
                    color = COLORS.get(t, COLORS[FLOOR])
                    pygame.draw.rect(screen, color, (rx, ry, tile - 1, tile - 1))

                    center = (rx + tile // 2, ry + tile // 2)
                    small = tile // 4

                    if t == KEY:
                        pygame.draw.circle(screen, (255, 215, 0), center, small)
                    elif t == TRAP:
                        pygame.draw.line(screen, (255, 255, 255), (rx + 4, ry + 4), (rx + tile - 5, ry + tile - 5), 2)
                        pygame.draw.line(screen, (255, 255, 255), (rx + tile - 5, ry + 4), (rx + 4, ry + tile - 5), 2)
                    elif t == GOAL:
                        pygame.draw.polygon(screen, (255, 255, 255), [
                            (center[0], ry + 3),
                            (rx + tile - 4, ry + tile - 4),
                            (rx + 3, ry + tile - 4),
                        ])
                    elif t == DOOR:
                        pygame.draw.rect(screen, (200, 150, 50), (rx + 2, ry + 2, tile - 5, tile - 5), 2)

                    pygame.draw.rect(screen, (50, 50, 70), (rx, ry, tile, tile), 1)

            # Enemies
            for er, ec in state.enemy_positions:
                ex = gx + ec * tile + tile // 2
                ey = gy + er * tile + tile // 2
                pygame.draw.circle(screen, COLORS[ENEMY], (ex, ey), tile // 3)

            # Player
            pr, pc = state.player_pos
            px = gx + pc * tile + tile // 2
            py = gy + pr * tile + tile // 2
            pygame.draw.circle(screen, COLORS["player"], (px, py), tile // 3)
            pygame.draw.circle(screen, (255, 255, 255), (px, py), tile // 3, 2)

            # HUD
            info_x = gx + state.cols * tile + 20
            texts = [
                f"Steps: {state.steps}",
                f"HP: {state.health}",
                f"Keys: {len(state.keys_collected)}/{state.total_keys}",
                f"Reward: {total_reward:.0f}",
                "",
                "WASD / Arrows = move",
                "Space = wait",
                "R = restart",
                "Esc = quit",
            ]
            if state.won:
                texts.insert(0, "YOU WON!")
            elif state.done:
                texts.insert(0, "GAME OVER")

            for i, text in enumerate(texts):
                color = (0, 255, 100) if "WON" in text else ((255, 60, 60) if "OVER" in text else COLORS["text"])
                surf = font.render(text, True, color)
                screen.blit(surf, (info_x, gy + i * 22))

            # Title
            title = font.render("AI Puzzle — Manual Play", True, COLORS["text"])
            screen.blit(title, (20, 20))

            pygame.display.flip()

    pygame.quit()
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Puzzle Solver Showdown — Classical vs RL vs LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Grid size (default: 9)")
    common.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Difficulty 1-5")
    common.add_argument("--seed", type=int, default=None, help="Random seed")
    common.add_argument("--llm-provider", type=str, default="claude", choices=["claude", "openai", "groq", "ollama"],
                        help="LLM provider: claude or openai")
    common.add_argument("--llm-model", type=str, default=None, help="Specific LLM model name")
    common.add_argument("--rl-snapshot", type=str, default=None,
                        help="Path to a specific RL snapshot (e.g. models/snapshots/dqn_grid9_d1_ep500.pt)")
    common.add_argument("--simple", action="store_true",
                        help="Goal only — no keys, traps, or enemies")

    # demo
    p_demo = sub.add_parser("demo", parents=[common], help="Visual solver comparison")
    p_demo.add_argument("--solvers", nargs="+", default=["astar", "rl", "llm"],
                        choices=["astar", "astar-safe", "bfs", "rl", "llm"],
                        help="Solvers to compare")
    p_demo.set_defaults(func=cmd_demo)

    # benchmark
    p_bench = sub.add_parser("benchmark", parents=[common], help="Run benchmark suite")
    p_bench.add_argument("--solvers", nargs="+", default=["astar", "bfs", "rl", "llm"],
                         choices=["astar", "astar-safe", "bfs", "rl", "llm"])
    p_bench.add_argument("--num-levels", type=int, default=20, help="Number of levels to test")
    p_bench.add_argument("--save-plot", type=str, default=None, help="Save dashboard to file")
    p_bench.set_defaults(func=cmd_benchmark)

    # train
    p_train = sub.add_parser("train", parents=[common], help="Train RL agent")
    p_train.add_argument("--episodes", type=int, default=2000, help="Training episodes")
    p_train.add_argument("--save-every", type=int, default=100, help="Save every N episodes")
    p_train.add_argument("--snapshot-every", type=int, default=500, help="Save snapshot every N episodes")
    p_train.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning, train on fixed difficulty")
    p_train.set_defaults(func=cmd_train)

    # play
    p_play = sub.add_parser("play", parents=[common], help="Play manually")
    p_play.set_defaults(func=cmd_play)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
