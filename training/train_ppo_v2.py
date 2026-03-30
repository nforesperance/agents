"""PPO v2 — larger network + curriculum + shorter episodes for small grids.

This is the best-effort training to reach A*-level on 10x10 mazes.
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from solvers.ppo_solver import PuzzleEnv

# (grid_size, num_traps, max_steps)
STAGES = [
    (5, 0, 30),
    (7, 0, 60),
    (8, 0, 100),
    (9, 0, 150),
    (10, 0, 200),
]

ADVANCE_THRESHOLD = 0.50
EVAL_EPISODES = 100
STEPS_PER_EVAL = 100_000
OBS_SIZE = 10


def make_env(grid_size, num_traps, max_steps):
    def _init():
        return PuzzleEnv(grid_size=grid_size, num_traps=num_traps, obs_size=OBS_SIZE, max_steps=max_steps)
    return _init


def evaluate(model, grid_size, num_traps, max_steps):
    env = PuzzleEnv(grid_size=grid_size, num_traps=num_traps, obs_size=OBS_SIZE, max_steps=max_steps)
    wins = 0
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, t, info = env.step(a)
            done = done or t
        if info.get("won"):
            wins += 1
    return wins / EVAL_EPISODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--n-envs", type=int, default=16)
    args = parser.parse_args()

    save_dir = args.save_dir

    # Guard: if save_dir looks like a Google Drive path, check it's mounted
    if "/content/drive" in save_dir:
        if not os.path.ismount("/content/drive"):
            print("ERROR: Google Drive not mounted! Run this first:")
            print("  from google.colab import drive; drive.mount('/content/drive')")
            sys.exit(1)
        # Verify we can actually write to Drive
        test_file = os.path.join(save_dir, ".write_test")
        os.makedirs(save_dir, exist_ok=True)
        try:
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except Exception as e:
            print(f"ERROR: Cannot write to {save_dir}: {e}")
            sys.exit(1)
    else:
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "snapshots"), exist_ok=True)

    model_path = os.path.join(save_dir, "ppo_v2_grid10.zip")
    state_path = os.path.join(save_dir, "ppo_v2_grid10_state.json")
    best_path = os.path.join(save_dir, "ppo_grid10_best.zip")

    # Load state
    stage = 0
    steps_done = 0
    best_solve = 0.0
    has_state = os.path.exists(state_path)
    has_model = os.path.exists(model_path)

    if has_state and not has_model:
        print(f"WARNING: State file found ({state_path}) but model is missing ({model_path})!")
        print("The checkpoint may be corrupted. Training will start from scratch.")
    elif not has_state and has_model:
        print(f"WARNING: Model found but no state file. Will load model but start at stage 0.")

    if has_state:
        with open(state_path) as f:
            st = json.load(f)
            stage = st.get("stage", 0)
            steps_done = st.get("steps_done", 0)
            best_solve = st.get("best_solve", 0.0)
        print(f"Resuming: stage {stage}, {steps_done:,} steps, best {best_solve*100:.0f}%")

    grid_size, num_traps, max_steps = STAGES[min(stage, len(STAGES) - 1)]

    print(f"PPO v2 — Larger network [512,512,256] + curriculum")
    print(f"Target: 10x10 no traps | Steps: {args.steps:,} | Envs: {args.n_envs}")
    print(f"Stage {stage}: {grid_size}x{grid_size} (max {max_steps} steps)")
    print("-" * 60)

    # Create env for current stage
    env = DummyVecEnv([make_env(grid_size, num_traps, max_steps) for _ in range(args.n_envs)])

    # Create or load model
    if has_model:
        model = PPO.load(model_path, env=env, device="cpu")
        print(f"Loaded {model_path}")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            verbose=0,
            device="cpu",
            policy_kwargs=dict(net_arch=[512, 512, 256]),
        )
        print("New model (512-512-256)")

    start = time.time()
    remaining = args.steps - steps_done

    while remaining > 0:
        grid_size, num_traps, max_steps = STAGES[min(stage, len(STAGES) - 1)]

        # Update envs
        env = DummyVecEnv([make_env(grid_size, num_traps, max_steps) for _ in range(args.n_envs)])
        model.set_env(env)

        # Train chunk
        model.learn(total_timesteps=STEPS_PER_EVAL, reset_num_timesteps=False)
        steps_done += STEPS_PER_EVAL
        remaining -= STEPS_PER_EVAL

        # Evaluate
        solve_rate = evaluate(model, grid_size, num_traps, max_steps)
        elapsed = time.time() - start

        print(
            f"Step {steps_done:>10,} [S{stage} {grid_size}x{grid_size}] | "
            f"Solved: {solve_rate*100:5.1f}% | "
            f"Time: {elapsed:.0f}s"
        )

        # Save + verify
        model.save(model_path)
        with open(state_path, "w") as f:
            json.dump({"stage": stage, "steps_done": steps_done, "best_solve": best_solve}, f)
        # Verify checkpoint actually landed on disk
        saved_ok = os.path.exists(model_path) or os.path.exists(model_path + ".zip")
        if not saved_ok:
            print(f"ERROR: Save failed — {model_path} not found after save!")
            print("Check that Google Drive is mounted.")
            sys.exit(1)

        if solve_rate > best_solve:
            best_solve = solve_rate
            model.save(best_path)
            print(f"  New best! {solve_rate*100:.1f}%")

        # Advance
        if solve_rate >= ADVANCE_THRESHOLD and stage < len(STAGES) - 1:
            snap = os.path.join(save_dir, "snapshots", f"ppo_v2_stage{stage}_done.zip")
            model.save(snap)
            stage += 1
            print(f"\n>>> ADVANCING to Stage {stage}: {STAGES[stage][0]}x{STAGES[stage][0]} "
                  f"(max {STAGES[stage][2]} steps)\n")
            with open(state_path, "w") as f:
                json.dump({"stage": stage, "steps_done": steps_done, "best_solve": best_solve}, f)

        if best_solve >= 0.90:
            print("TARGET 90% REACHED!")
            break

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Best: {best_solve*100:.1f}%. Stage: {stage}/{len(STAGES)-1}")


if __name__ == "__main__":
    main()
