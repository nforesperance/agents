de recherche : BFS, DFS, A*, et applications comme la planification (séquentielle) et la résolution de puzzles. Processus de décision markoviens (MDP). Apprentissage par renforcement (RL).
Semaine 6 18 février 2026 Introduction à l’apprentissage machine Apprentissage supervisé et non supervisé, avec des algorithmes classiques (régression, k-NN, etc.).
Semaine 7 25 février 2026 Réseaux de neurones et apprentissage profond Neurones artificiels, architectures MLP, CNN, RNN, et introduction aux frameworks comme TensorFlow et PyTorch.
Pause 4 mars 2026 Pause (Relâche) Pas de cours cette semaine.
Semaine 8 11 mars 2026 IA générative GANs, VAEs, modèles de diffusion, applications créatives, enjeux éthiques et perspectives.
Semaine 9 18 mars 2026 Langage naturel, compréhension et LLMs Introduction au traitement automatique des langues (TAL). Présentation des LLMs : architecture Transformer, pré-entraînement, fine-tuning. Applications : traduction, chatbots, génération de contenu. Défis : biais, ambiguïté, hallucination.
Semaine 10 25 mars 2026 Agents intelligents Concepts fondamentaux, historique, types d’agents (réactifs, basés sur des objectifs, utilitaires), et exemples d’applications.
Semaine 11 1 avril 2026 Présentations personnelles (1/2) Premières présentations individuelles sur des sujets liés aux concepts abordés dans le cours.
Semaine 12 8 avril 2026 Présentations personnelles (2/2) Deuxièmes présentations individuelles sur des sujets liés aux concepts abordés dans le cours.
Semaine 13 15 avril 2026 [IA opérationnelle] IA embarquée, conception et déploiement de solutions IA.
Semaine 14 22 avril 2026 Examen final Évaluation écrite sur l’ensemble des notions abordées dans le cours.
Semaine 15 29 avril 2026 Présentations en équipes Présentations en groupes sur des projets pratiques intégrant plusieurs concepts vus dans le cours.
Modalités d’évaluation
Description sommaire Date Pondération
Examen écrit final Fin de session 40%
Présentation personnelle En cours de session 30%
Projet et présentation en équipe En cours de session 30%
uqàm

Copyright (c)Laurent Magnin / UQÀM 2025-2026
```
```
Option A: AI Arena — Agents Learn to Fight & Survive
A 2D arena where RL agents learn combat strategies from scratch. You watch them go from random movement to intelligent behavior.

Custom 2D game (Pygame) — agents have health, energy, attacks, dodge
RL training (PPO/DQN) — agents learn entirely through trial and error
Multiple agent species with different reward functions (aggressive, defensive, cooperative)
Emergent behavior — alliances, ambushes, flee strategies appear naturally
LLM commentator narrates the fights in real-time like a sports broadcast
Evolution mode — genetic algorithm selects best agents across generations
Covers: RL, neural networks, search, agent types, generative AI (commentary)
Option B: Ecosystem Simulation — Creatures That Learn to Live
A living 2D world where creatures evolve neural network "brains" and learn to survive.

Predators & prey with vision, hunger, reproduction
Neural network brains (small MLPs) controlling each creature
RL + genetic algorithms — creatures learn within a lifetime (RL) AND across generations (evolution)
Emergent behaviors — herding, hunting packs, camouflage strategies
LLM naturalist — an AI David Attenborough that describes what's happening
Dashboard with population graphs, fitness curves, species trees
Covers: RL, neural networks, genetic algorithms, agents, NLP
Option C: AI Learns Your Game — Procedural Puzzle Solver Showdown
Build a custom puzzle game, then pit 3 AI paradigms against each other to solve it live.

Custom roguelike puzzle game (grid-based, traps, keys, enemies, goals)
3 solvers visualized side-by-side:
Classical AI (A*, BFS) — perfect but slow
RL agent (DQN/PPO) — learns through experience
LLM agent (Claude) — reasons about the puzzle in natural language
Real-time visualization — watch each approach think differently
Benchmark dashboard — speed, success rate, generalization to new levels
Procedural generation — infinite levels to test generalization
Covers: search algorithms, RL, neural networks, LLMs, agents, planning
My recommendation: Option C. It directly compares classical AI vs deep RL vs LLMs on the same task — that's a thesis-worthy comparison framework, visually spectacular, and tells a clear story about the evolution of AI approaches.




Game Rules
Objective: Navigate from your start position to the Goal tile to win.

Tiles
Color	Symbol	Meaning
Green	G	Goal — reach this to win
Gold/Yellow	K	Key — collect ALL keys before you can open doors
Brown	D	Door — blocks your path until you have all keys
Red (dark)	X	Trap — deals 25 HP damage (triggers once then disappears)
Magenta/Pink	E	Enemy — patrols back and forth, deals 30 HP damage on contact
Light blue	S	Start — your spawn point
Cyan circle	P	You (the player)
Dark gray	#	Wall — impassable
Medium gray	.	Floor — safe to walk on
Mechanics
HP: You start at 100. Traps cost 25, enemies cost 30. At 0 HP = Game Over
Keys & Doors: You must collect all keys on the level before any door opens
Enemies: They patrol in a line, reversing direction when hitting a wall
Step limit: 200 moves max, then you time out
Reward: -1 per step, +10 per key, +100 for reaching goal, -10 for traps, -20 for enemies
Controls
WASD or Arrow keys = move
Space = wait (skip a turn)
R = restart the level
Esc = quit


project/
├── main.py                    # CLI entry point
├── config.py                  # All settings & constants
├── requirements.txt
├── game/
│   ├── engine.py              # Game state, actions, rewards
│   └── level_generator.py     # Procedural maze generation (Prim's)
├── solvers/
│   ├── base.py                # Solver interface
│   ├── classical.py           # A* and BFS (with key-aware state space)
│   ├── rl_solver.py           # DQN with CNN, replay buffer, training
│   └── llm_solver.py          # Claude AND ChatGPT support
├── training/
│   └── train_rl.py            # RL training loop with curves
├── ui/
│   ├── visualizer.py          # Pygame side-by-side view
│   └── dashboard.py           # Matplotlib benchmark charts
├── benchmarks/
│   └── runner.py              # Multi-level evaluation
└── models/                    # Saved RL weights



source venv/bin/activate

# 1. Play manually first
python main.py play

# 2. Train the RL agent
python main.py train --episodes 2000 --difficulty 1

# 3. Watch solvers compete side-by-side
python main.py demo --solvers astar rl llm

# 4. Use ChatGPT instead of Claude
python main.py demo --llm-provider openai

# 5. Full benchmark with charts
python main.py benchmark --num-levels 20 --difficulty 2




What each solver demonstrates (course mapping)
Solver	Course Weeks
A*/BFS	Week 5 — Search algorithms, planning
DQN	Week 5 (RL, MDP), Week 6 (ML), Week 7 (neural nets)
LLM	Week 9 (NLP, Transformers, LLMs)
The framework itself	Week 10 (agent architectures), Week 13 (deployment)
Next steps
You'll want to train the RL agent first (python main.py train) so it actually learns, then run the demo. Want me to kick off a training run, or would you like to modify anything first?