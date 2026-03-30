"""Streamlit UI for AI Puzzle Solver Showdown.

Launch with: streamlit run app.py
"""

import streamlit as st
import subprocess
import os

st.set_page_config(page_title="AI Puzzle Solver Showdown", layout="wide")

st.title("AI Puzzle Solver Showdown")
st.markdown("*Comparaison de paradigmes d'IA sur un jeu de puzzle procedural*")

st.divider()

# --- Sidebar: common settings ---
st.sidebar.header("Settings")

grid_size = st.sidebar.slider("Grid size", 5, 30, 9)
difficulty = st.sidebar.slider("Difficulty", 1, 5, 1)
seed = st.sidebar.number_input("Seed (0 = random)", 0, 999999, 0)
simple = st.sidebar.checkbox("Simple (goal only, no traps/keys/enemies)", value=False)

seed_arg = f"--seed {seed}" if seed > 0 else ""
simple_arg = "--simple" if simple else ""

st.divider()

# --- Mode selection ---
mode = st.radio(
    "Mode",
    ["Demo (side-by-side solvers)", "Play (manual)", "Train RL (DQN)", "Train RL (PPO)", "Benchmark"],
    horizontal=True,
)

if mode == "Demo (side-by-side solvers)":
    st.subheader("Demo — Solver Comparison")

    col1, col2 = st.columns(2)
    with col1:
        solvers = st.multiselect(
            "Solvers",
            ["astar", "astar-safe", "bfs", "rl", "rl2", "llm"],
            default=["astar", "astar-safe"],
        )
    with col2:
        llm_provider = st.selectbox("LLM Provider", ["ollama", "groq", "openai", "claude"])
        rl_snapshot = st.text_input("RL snapshot path (optional)", "")

    solvers_arg = " ".join(solvers)
    llm_arg = f"--llm-provider {llm_provider}" if "llm" in solvers else ""
    snap_arg = f"--rl-snapshot {rl_snapshot}" if rl_snapshot else ""

    cmd = f"python main.py demo --solvers {solvers_arg} --grid-size {grid_size} --difficulty {difficulty} {seed_arg} {simple_arg} {llm_arg} {snap_arg}"

    st.code(cmd.strip(), language="bash")
    if st.button("Launch Demo", type="primary"):
        st.info("Launching Pygame window...")
        subprocess.Popen(cmd.strip().split())


elif mode == "Play (manual)":
    st.subheader("Play — Manual Mode")

    cmd = f"python main.py play --grid-size {grid_size} --difficulty {difficulty} {seed_arg} {simple_arg}"

    st.code(cmd.strip(), language="bash")
    if st.button("Launch Game", type="primary"):
        st.info("Launching Pygame window...")
        subprocess.Popen(cmd.strip().split())


elif mode == "Train RL (DQN)":
    st.subheader("Train — DQN Agent")

    episodes = st.number_input("Episodes", 1000, 100000, 10000, step=1000)
    snapshot_every = st.number_input("Snapshot every N episodes", 100, 5000, 500)
    no_curriculum = st.checkbox("Disable curriculum")

    curr_arg = "--no-curriculum" if no_curriculum else ""
    cmd = f"python main.py train --episodes {episodes} --grid-size {grid_size} --difficulty {difficulty} --snapshot-every {snapshot_every} {curr_arg}"

    st.code(cmd.strip(), language="bash")
    if st.button("Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            result = subprocess.run(cmd.strip().split(), capture_output=True, text=True, timeout=3600)
            st.text(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
            if result.returncode != 0:
                st.error(result.stderr[-1000:])


elif mode == "Train RL (PPO)":
    st.subheader("Train — PPO Agent")

    total_steps = st.number_input("Total steps", 100000, 2000000, 500000, step=100000)
    num_traps = st.slider("Number of traps", 0, 30, 0)
    n_envs = st.slider("Parallel environments", 1, 16, 8)
    save_dir = st.text_input("Save directory", "models")
    snapshot_every = st.number_input("Snapshot every N steps", 10000, 200000, 50000, step=10000)

    cmd = f"python training/train_ppo.py --steps {total_steps} --grid-size {grid_size} --traps {num_traps} --n-envs {n_envs} --save-dir {save_dir} --snapshot-every {snapshot_every}"

    st.code(cmd.strip(), language="bash")
    if st.button("Start PPO Training", type="primary"):
        with st.spinner("Training in progress..."):
            result = subprocess.run(cmd.strip().split(), capture_output=True, text=True, timeout=7200)
            st.text(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
            if result.returncode != 0:
                st.error(result.stderr[-1000:])


elif mode == "Benchmark":
    st.subheader("Benchmark — Multi-level Evaluation")

    col1, col2 = st.columns(2)
    with col1:
        solvers = st.multiselect(
            "Solvers to benchmark",
            ["astar", "astar-safe", "bfs", "rl", "rl2"],
            default=["astar", "astar-safe"],
        )
    with col2:
        num_levels = st.number_input("Number of levels", 5, 100, 20)
        save_plot = st.text_input("Save plot to (optional)", "")

    solvers_arg = " ".join(solvers)
    plot_arg = f"--save-plot {save_plot}" if save_plot else ""

    cmd = f"python main.py benchmark --solvers {solvers_arg} --grid-size {grid_size} --difficulty {difficulty} --num-levels {num_levels} {seed_arg} {plot_arg}"

    st.code(cmd.strip(), language="bash")
    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running benchmark..."):
            result = subprocess.run(cmd.strip().split(), capture_output=True, text=True, timeout=600)
            st.text(result.stdout)
            if save_plot and os.path.exists(save_plot):
                st.image(save_plot)


# --- Footer ---
st.divider()

# Show available models
st.sidebar.divider()
st.sidebar.subheader("Available Models")
for root, dirs, files in os.walk("models"):
    for f in sorted(files):
        if f.endswith((".pt", ".zip")):
            path = os.path.join(root, f)
            size = os.path.getsize(path) / 1024 / 1024
            st.sidebar.text(f"{path} ({size:.1f}MB)")

if not any(f.endswith((".pt", ".zip")) for _, _, files in os.walk("models") for f in files):
    st.sidebar.text("No models found. Train one first!")
