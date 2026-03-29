# AI Puzzle Solver Showdown

**Comparaison de paradigmes d'IA sur un jeu de puzzle procedural**

Projet de session — INF7370 / UQAM 2025-2026

---

## Description

Ce projet compare **quatre paradigmes d'intelligence artificielle** sur une meme tache :
resoudre des niveaux de puzzle generes proceduralement.

Les solveurs s'affrontent cote a cote en temps reel, permettant de visualiser
comment chaque approche "pense" differemment.

| Solveur | Paradigme | Concepts du cours |
|---------|-----------|-------------------|
| **BFS** | Recherche en largeur | Semaine 5 — Algorithmes de recherche |
| **A\*** | Recherche informee (heuristique) | Semaine 5 — A*, planification |
| **A\* Safe** | Recherche avec couts de danger | Semaine 5 — A* avec fonction de cout |
| **DQN (RL)** | Apprentissage par renforcement | Semaines 5, 6, 7 — MDP, RL, reseaux de neurones |
| **LLM** | Modele de langage (Claude / ChatGPT) | Semaine 9 — TAL, Transformers, LLMs |

Le cadre de comparaison lui-meme illustre les concepts de la **Semaine 10**
(agents intelligents : reactifs, bases sur objectifs, utilitaires) et de la
**Semaine 13** (deploiement de solutions IA).

---

## Regles du jeu

Le joueur doit naviguer dans une grille pour atteindre l'objectif (Goal).

### Tuiles

| Couleur | Symbole | Signification |
|---------|---------|---------------|
| Vert | G | **Goal** — atteindre pour gagner |
| Jaune/Or | K | **Key** — collecter TOUTES les cles avant d'ouvrir les portes |
| Brun | D | **Door** — bloque le passage tant que toutes les cles ne sont pas collectees |
| Rouge fonce | X | **Trap** — inflige 25 HP de degats (se declenche une seule fois) |
| Magenta/Rose | E | **Enemy** — patrouille d'avant en arriere, inflige 30 HP de degats au contact |
| Bleu clair | S | **Start** — point de depart |
| Cercle cyan | P | **Player** — le joueur |
| Gris fonce | # | **Wall** — infranchissable |
| Gris moyen | . | **Floor** — sol sur |

### Mecanique

- **HP (Health Points)** : Le joueur commence a 100 HP. Pieges = -25 HP, Ennemis = -30 HP. A 0 HP = Game Over.
- **Cles et portes** : Toutes les cles du niveau doivent etre collectees avant de pouvoir franchir une porte.
- **Ennemis** : Ils patrouillent en ligne droite, inversant leur direction lorsqu'ils frappent un mur.
- **Limite de pas** : 200 mouvements maximum, apres quoi c'est un timeout.
- **Recompenses** : -1 par pas, +10 par cle, +100 pour le goal, -10 pour les pieges, -20 pour les ennemis.

### Controles (mode Play)

- **WASD** ou **fleches directionnelles** = deplacer
- **Espace** = attendre (sauter un tour)
- **R** = recommencer le niveau
- **Esc** = quitter

---

## Architecture du projet

```
project/
├── main.py                    # Point d'entree CLI (demo, play, train, benchmark)
├── config.py                  # Configuration globale (tailles, couleurs, parametres RL)
├── requirements.txt           # Dependances Python
├── game/
│   ├── engine.py              # Moteur de jeu : etat, actions, regles, recompenses
│   └── level_generator.py     # Generation procedurale de niveaux (algorithme de Prim)
├── solvers/
│   ├── base.py                # Interface commune pour tous les solveurs
│   ├── classical.py           # A*, A* Safe (cost-aware), BFS
│   ├── rl_solver.py           # DQN avec CNN (PyTorch), replay buffer, entrainement
│   └── llm_solver.py          # Solveur LLM (Claude API + OpenAI API)
├── training/
│   └── train_rl.py            # Boucle d'entrainement RL avec courbes d'apprentissage
├── ui/
│   ├── visualizer.py          # Visualisation Pygame cote-a-cote des solveurs
│   └── dashboard.py           # Tableau de bord de benchmarks (Matplotlib)
├── benchmarks/
│   └── runner.py              # Evaluation multi-niveaux
└── models/                    # Poids du modele RL entraines
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Cles API (pour le solveur LLM)

```bash
# Pour utiliser Claude :
export ANTHROPIC_API_KEY="votre-cle"

# Pour utiliser ChatGPT :
export OPENAI_API_KEY="votre-cle"
```

---

## Utilisation

### 1. Jouer manuellement

```bash
python main.py play
python main.py play --difficulty 3 --seed 42
```

### 2. Entrainer l'agent RL

```bash
python main.py train --episodes 2000 --difficulty 1
python main.py train --episodes 5000 --difficulty 3 --grid-size 11
```

L'entrainement sauvegarde les poids dans `models/` et affiche les courbes
d'apprentissage (recompense, taux de succes, perte).

### 3. Demo visuelle (comparaison cote-a-cote)

```bash
# A* vs BFS (classique)
python main.py demo --solvers astar bfs

# A* vs A* Safe (impact de la conscience des dangers)
python main.py demo --solvers astar astar-safe --difficulty 3

# Tous les solveurs
python main.py demo --solvers astar astar-safe rl llm

# Utiliser ChatGPT au lieu de Claude
python main.py demo --solvers astar rl llm --llm-provider openai

# Reproduire un niveau specifique
python main.py demo --seed 78190 --difficulty 4 --solvers astar astar-safe
```

### 4. Benchmark complet

```bash
python main.py benchmark --num-levels 20 --difficulty 2
python main.py benchmark --solvers astar astar-safe rl --num-levels 50 --save-plot results.png
```

### Options communes

| Option | Description | Defaut |
|--------|-------------|--------|
| `--grid-size` | Taille de la grille | 9 |
| `--difficulty` | Difficulte 1-5 (affecte taille, pieges, ennemis, cles) | 1 |
| `--seed` | Graine aleatoire (pour reproduire un niveau) | aleatoire |
| `--llm-provider` | `claude` ou `openai` | `claude` |
| `--llm-model` | Modele LLM specifique | dernier disponible |

### Solveurs disponibles

| Cle CLI | Nom | Description |
|---------|-----|-------------|
| `astar` | A* | Plus court chemin, ignore les dangers |
| `astar-safe` | A* Safe | Chemin optimal en tenant compte des pieges et ennemis |
| `bfs` | BFS | Recherche en largeur, plus court chemin garanti |
| `rl` | RL (DQN) | Agent entraine par renforcement (necessite entrainement prealable) |
| `llm` | LLM | Raisonnement en langage naturel (Claude ou ChatGPT) |

---

## Description des solveurs

### BFS (Breadth-First Search)

Explore tous les etats couche par couche. Garantit le chemin le plus court
en nombre de pas mais n'a aucune conscience des dangers. Sert de base de
comparaison.

**Complexite** : O(V + E) ou V = nombre d'etats, E = nombre de transitions.

### A* (A-Star)

Recherche informee avec heuristique de distance de Manhattan. Plus efficace
que BFS grace a l'heuristique admissible, mais ignore egalement les dangers
(pieges, ennemis). Trouve le chemin optimal en nombre de pas.

**Heuristique** : Distance de Manhattan vers la cle non collectee la plus
proche + distance de cette cle vers l'objectif.

### A* Safe (Cost-Aware A*)

Variante de A* qui attribue des couts supplementaires aux tuiles dangereuses :
- Pieges : +10 au cout de deplacement
- Cases adjacentes aux ennemis : +5 au cout de deplacement

Produit des chemins plus longs mais plus surs. Illustre comment l'ajout d'une
fonction de cout modifie le comportement de recherche.

### DQN (Deep Q-Network)

Agent d'apprentissage par renforcement utilisant un reseau de neurones
convolutionnel (CNN) pour approximer la fonction Q.

**Architecture** :
- Entree : grille 5 canaux (tuiles, joueur, ennemis, cles, HP)
- 3 couches Conv2D (32, 64, 64 filtres)
- 2 couches FC (256, 128)
- Sortie : 5 valeurs Q (UP, DOWN, LEFT, RIGHT, WAIT)

**Entrainement** :
- Experience Replay (buffer de 50 000 transitions)
- Target Network (mise a jour toutes les 10 episodes)
- Epsilon-greedy avec decroissance (1.0 -> 0.05)

### LLM (Large Language Model)

Le solveur LLM envoie une representation textuelle de la grille a un modele
de langage (Claude ou ChatGPT) et lui demande de choisir la prochaine action.

Le modele recoit :
- La grille sous forme de texte avec legende
- L'etat courant (HP, cles collectees, pas restants)
- Un prompt systeme decrivant les regles et la strategie

Supporte **Claude** (API Anthropic) et **ChatGPT** (API OpenAI), selectionnable
a l'execution via `--llm-provider`.

---

## Generation procedurale des niveaux

Les niveaux sont generes avec l'**algorithme de Prim randomise** :

1. Initialiser une grille pleine de murs
2. Creuser des passages avec l'algorithme de Prim
3. Ouvrir des passages supplementaires pour plus de variete
4. Placer depart, objectif (le plus loin possible), cles, porte, pieges, ennemis
5. Verifier la solvabilite par BFS

La difficulte (1-5) ajuste :
- Taille de la grille (9 -> 15)
- Nombre de cles (1 -> 4)
- Nombre de pieges (3 -> 13)
- Nombre d'ennemis (1 -> 4)

---

## Controles de la visualisation (mode Demo)

| Touche | Action |
|--------|--------|
| **Espace** | Pause / reprendre |
| **Fleche droite** | Avancer d'un pas |
| **Fleche haut** | Accelerer |
| **Fleche bas** | Ralentir |
| **R** | Recommencer |
| **Q / Esc** | Quitter |

---

## Correspondance avec le cours

| Composante du projet | Semaine(s) du cours |
|---------------------|---------------------|
| BFS, A*, A* Safe | Semaine 5 — Algorithmes de recherche, planification |
| DQN, MDP, recompenses | Semaine 5 — MDP et apprentissage par renforcement |
| Reseau de neurones CNN | Semaine 7 — Reseaux de neurones et apprentissage profond |
| Solveur LLM (Claude/GPT) | Semaine 9 — TAL, Transformers, LLMs |
| Architecture d'agents | Semaine 10 — Agents intelligents |
| Benchmark et deploiement | Semaine 13 — IA operationnelle |

---

## Credits

- **Generation de niveaux** : Algorithme de Prim randomise
- **Visualisation** : Pygame
- **RL** : PyTorch (DQN)
- **LLM** : API Anthropic (Claude) / API OpenAI (ChatGPT)
- **Benchmarks** : Matplotlib

Projet realise dans le cadre du cours INF7370 — Laurent Magnin / UQAM 2025-2026.
