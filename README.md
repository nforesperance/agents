# AI Puzzle Solver Showdown

**Comparaison de paradigmes d'IA sur un jeu de puzzle procedural**

Projet de session -- INF8790 Fondements de l'Intelligence Artificielle / UQAM 2025-2026

---

## Equipe

| Membre | GitHub |
|--------|--------|
| Membre 1 | [@github_id1](https://github.com/github_id1) |
| Membre 2 | [@github_id2](https://github.com/github_id2) |
| Membre 3 | [@github_id3](https://github.com/github_id3) |

---

## Table des matieres

- [Description](#description)
- [Technologies IA utilisees](#technologies-ia-utilisees)
- [Regles du jeu](#regles-du-jeu)
- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Description des solveurs](#description-des-solveurs)
- [Generation procedurale des niveaux](#generation-procedurale-des-niveaux)
- [Entrainement RL (curriculum)](#entrainement-rl-curriculum)
- [Benchmarks](#benchmarks)
- [Correspondance avec le cours](#correspondance-avec-le-cours)
- [Blog du projet](#blog-du-projet)
- [Contribuer](#contribuer)
- [Licence](#licence)
- [Credits](#credits)

---

## Description

Ce projet compare **cinq solveurs** issus de **trois paradigmes d'intelligence artificielle** sur une meme tache : resoudre des niveaux de puzzle generes proceduralement.

Les solveurs s'affrontent cote a cote en temps reel, permettant de visualiser comment chaque approche "pense" differemment. Le cadre d'evaluation comprend un benchmark multi-niveaux avec tableau de bord graphique, un mode demo visuel Pygame, et un mode jeu manuel.

---

## Technologies IA utilisees

| Solveur | Paradigme | Concepts du cours |
|---------|-----------|-------------------|
| **BFS** | Recherche en largeur | Semaine 5 -- Algorithmes de recherche |
| **A\*** | Recherche informee (heuristique) | Semaine 5 -- A*, planification |
| **A\* Safe** | Recherche avec couts de danger | Semaine 5 -- A* avec fonction de cout |
| **DQN (RL)** | Apprentissage par renforcement | Semaines 5, 6, 7 -- MDP, RL, reseaux de neurones (CNN) |
| **LLM** | Modele de langage (Claude / ChatGPT) | Semaine 9 -- TAL, Transformers, LLMs |

Le cadre de comparaison illustre egalement les concepts de la **Semaine 10** (agents intelligents : reactifs, bases sur objectifs, utilitaires) et de la **Semaine 13** (deploiement et operationnalisation de solutions IA).

---

## Regles du jeu

Le joueur doit naviguer dans une grille pour atteindre l'objectif (Goal).

### Tuiles

| Couleur | Symbole | Signification |
|---------|---------|---------------|
| Vert | G | **Goal** -- atteindre pour gagner |
| Jaune/Or | K | **Key** -- collecter TOUTES les cles avant d'ouvrir les portes |
| Brun | D | **Door** -- bloque le passage tant que toutes les cles ne sont pas collectees |
| Rouge fonce | X | **Trap** -- inflige 25 HP de degats (se declenche une seule fois) |
| Magenta/Rose | E | **Enemy** -- patrouille d'avant en arriere, inflige 30 HP de degats au contact |
| Bleu clair | S | **Start** -- point de depart |
| Cercle cyan | P | **Player** -- le joueur |
| Gris fonce | # | **Wall** -- infranchissable |
| Gris moyen | . | **Floor** -- sol sur |

### Mecanique

- **HP (Health Points)** : Le joueur commence a 100 HP. Pieges = -25 HP, Ennemis = -30 HP. A 0 HP = Game Over.
- **Cles et portes** : Toutes les cles du niveau doivent etre collectees avant de pouvoir franchir une porte.
- **Ennemis** : Ils patrouillent en ligne droite, inversant leur direction lorsqu'ils frappent un mur.
- **Limite de pas** : 200 mouvements maximum, apres quoi c'est un timeout.
- **Recompenses** : -1 par pas, +10 par cle, +100 pour le goal, -10 pour les pieges, -20 pour les ennemis, -50 pour le timeout.

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
├── docs/
│   └── blog.md                # Blog detaille du projet
├── game/
│   ├── engine.py              # Moteur de jeu : etat, actions, regles, recompenses
│   └── level_generator.py     # Generation procedurale de niveaux (algorithme de Prim)
├── solvers/
│   ├── base.py                # Interface commune pour tous les solveurs
│   ├── classical.py           # A*, A* Safe (cost-aware), BFS
│   ├── rl_solver.py           # DQN avec CNN (PyTorch), replay buffer, entrainement
│   └── llm_solver.py          # Solveur LLM (Claude API + OpenAI API)
├── training/
│   └── train_rl.py            # Boucle d'entrainement RL avec curriculum et courbes
├── ui/
│   ├── visualizer.py          # Visualisation Pygame cote-a-cote des solveurs
│   └── dashboard.py           # Tableau de bord de benchmarks (Matplotlib)
├── benchmarks/
│   └── runner.py              # Evaluation multi-niveaux avec metriques
└── models/                    # Poids du modele RL entraines
    └── snapshots/             # Instantanes periodiques pendant l'entrainement
```

---

## Installation

### Prerequis

- Python 3.10+
- pip

### Etapes

```bash
# 1. Cloner le depot
git clone https://github.com/<votre-organisation>/ai-puzzle-solver-showdown.git
cd ai-puzzle-solver-showdown

# 2. Creer et activer un environnement virtuel
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# 3. Installer les dependances
pip install -r requirements.txt
```

### Cles API (pour le solveur LLM)

```bash
# Pour utiliser Claude (Anthropic) :
export ANTHROPIC_API_KEY="votre-cle"

# Pour utiliser ChatGPT (OpenAI) :
export OPENAI_API_KEY="votre-cle"
```

> **Note** : Les cles API ne sont necessaires que pour le solveur LLM. Tous les autres solveurs fonctionnent sans configuration supplementaire.

---

## Utilisation

Le point d'entree unique est `main.py`, qui expose quatre sous-commandes :

### 1. Jouer manuellement (`play`)

```bash
python main.py play
python main.py play --difficulty 3 --seed 42
python main.py play --grid-size 11
```

### 2. Entrainer l'agent RL (`train`)

```bash
# Entrainement avec curriculum (recommande)
python main.py train --episodes 5000

# Entrainement sur une difficulte fixe (sans curriculum)
python main.py train --episodes 2000 --difficulty 3 --no-curriculum

# Parametres avances
python main.py train --episodes 10000 --grid-size 11 --save-every 200 --snapshot-every 1000
```

L'entrainement sauvegarde les poids dans `models/` et les instantanes dans `models/snapshots/`. Les courbes d'apprentissage (recompense, taux de succes, perte) sont affichees et sauvegardees en fin d'entrainement.

### 3. Demo visuelle (`demo`)

Visualisation cote-a-cote des solveurs en temps reel sur un meme niveau.

```bash
# Solveurs par defaut (A*, RL, LLM)
python main.py demo

# Choisir les solveurs
python main.py demo --solvers astar bfs
python main.py demo --solvers astar astar-safe --difficulty 3
python main.py demo --solvers astar astar-safe rl llm

# Utiliser ChatGPT au lieu de Claude
python main.py demo --solvers astar llm --llm-provider openai

# Reproduire un niveau avec une graine specifique
python main.py demo --seed 78190 --difficulty 4 --solvers astar astar-safe

# Utiliser un instantane RL specifique
python main.py demo --solvers astar rl --rl-snapshot models/snapshots/dqn_grid9_d1_ep500.pt
```

### 4. Benchmark complet (`benchmark`)

```bash
python main.py benchmark --num-levels 20 --difficulty 2
python main.py benchmark --solvers astar astar-safe rl --num-levels 50 --save-plot results.png
python main.py benchmark --solvers astar bfs rl llm --num-levels 10 --difficulty 3
```

---

## Reference CLI complete

### Options communes (disponibles pour toutes les sous-commandes)

| Option | Type | Defaut | Description |
|--------|------|--------|-------------|
| `--grid-size` | `int` | `9` | Taille de la grille (min 7, max 15) |
| `--difficulty` | `int` | `1` | Difficulte de 1 a 5 (affecte taille, pieges, ennemis, cles) |
| `--seed` | `int` | aleatoire | Graine aleatoire pour reproduire un niveau |
| `--llm-provider` | `str` | `claude` | Fournisseur LLM : `claude` ou `openai` |
| `--llm-model` | `str` | dernier | Modele LLM specifique (ex: `gpt-4o`, `claude-sonnet-4-20250514`) |
| `--rl-snapshot` | `str` | `None` | Chemin vers un instantane RL specifique |

### Options specifiques a `demo`

| Option | Type | Defaut | Description |
|--------|------|--------|-------------|
| `--solvers` | `str[]` | `astar rl llm` | Solveurs a comparer |

### Options specifiques a `benchmark`

| Option | Type | Defaut | Description |
|--------|------|--------|-------------|
| `--solvers` | `str[]` | `astar bfs rl llm` | Solveurs a evaluer |
| `--num-levels` | `int` | `20` | Nombre de niveaux a tester |
| `--save-plot` | `str` | `None` | Chemin pour sauvegarder le tableau de bord en image |

### Options specifiques a `train`

| Option | Type | Defaut | Description |
|--------|------|--------|-------------|
| `--episodes` | `int` | `2000` | Nombre d'episodes d'entrainement |
| `--save-every` | `int` | `100` | Sauvegarder un checkpoint tous les N episodes |
| `--snapshot-every` | `int` | `500` | Sauvegarder un instantane tous les N episodes |
| `--no-curriculum` | `flag` | `false` | Desactiver l'apprentissage par curriculum |

### Solveurs disponibles

| Cle CLI | Nom | Description |
|---------|-----|-------------|
| `astar` | A* | Plus court chemin avec heuristique Manhattan, ignore les dangers |
| `astar-safe` | A* Safe | Chemin optimal pondere tenant compte des pieges et ennemis |
| `bfs` | BFS | Recherche en largeur, plus court chemin garanti, ignore les dangers |
| `rl` | RL (DQN) | Agent entraine par renforcement (necessite entrainement prealable) |
| `llm` | LLM | Raisonnement en langage naturel via Claude ou ChatGPT |

---

## Description des solveurs

### BFS (Breadth-First Search)

Explore tous les etats couche par couche. Garantit le chemin le plus court en nombre de pas mais n'a aucune conscience des dangers. Sert de base de comparaison.

**Complexite** : O(V + E) ou V = nombre d'etats, E = nombre de transitions.

### A* (A-Star)

Recherche informee avec heuristique de distance de Manhattan. Plus efficace que BFS grace a l'heuristique admissible, mais ignore egalement les dangers (pieges, ennemis). Trouve le chemin optimal en nombre de pas.

**Heuristique** : Distance de Manhattan vers la cle non collectee la plus proche + distance de cette cle vers l'objectif.

### A* Safe (Cost-Aware A*)

Variante de A* qui attribue des couts supplementaires aux tuiles dangereuses :
- Pieges : +10 au cout de deplacement
- Cases adjacentes aux ennemis : +5 au cout de deplacement

Produit des chemins plus longs mais plus surs. Illustre comment l'ajout d'une fonction de cout modifie le comportement de recherche. L'heuristique reste admissible, garantissant l'optimalite par rapport au cout pondere.

### DQN (Deep Q-Network)

Agent d'apprentissage par renforcement utilisant un reseau de neurones convolutionnel (CNN) pour approximer la fonction Q.

**Architecture** :
- Entree : grille 5 canaux (tuiles, joueur, ennemis, cles, HP)
- 3 couches Conv2D (32, 64, 64 filtres, kernel 3x3, padding 1)
- 2 couches FC (256, 128)
- Sortie : 5 valeurs Q (UP, DOWN, LEFT, RIGHT, WAIT)

**Entrainement** :
- Experience Replay (buffer de 50 000 transitions)
- Target Network (mise a jour toutes les 10 episodes)
- Epsilon-greedy avec decroissance (1.0 -> 0.05, facteur 0.995)
- Apprentissage par curriculum en 6 etapes progressives
- Gradient clipping (max norm = 1.0)
- Fonction de perte : Smooth L1 (Huber)

### LLM (Large Language Model)

Le solveur LLM envoie une representation textuelle de la grille a un modele de langage (Claude ou ChatGPT) et lui demande de choisir la prochaine action a chaque pas.

Le modele recoit :
- La grille sous forme de texte avec legende
- L'etat courant (HP, cles collectees, pas restants)
- Un prompt systeme decrivant les regles et la strategie optimale
- L'historique des 10 derniers echanges (pour la coherence)

Supporte **Claude** (API Anthropic) et **ChatGPT** (API OpenAI), selectionnable a l'execution via `--llm-provider`.

---

## Generation procedurale des niveaux

Les niveaux sont generes avec l'**algorithme de Prim randomise** :

1. Initialiser une grille pleine de murs
2. Creuser des passages avec l'algorithme de Prim (a partir de la cellule 1,1)
3. Ouvrir des passages supplementaires (rows*cols/8) pour plus de variete topologique
4. Placer le depart, puis l'objectif (le plus loin possible par distance de Manhattan)
5. Placer les cles, la porte (pres du point median entre depart et objectif), les pieges et les ennemis
6. Verifier la solvabilite par BFS (avec et sans cles)

La difficulte (1-5) ajuste :

| Parametre | Diff. 1 | Diff. 2 | Diff. 3 | Diff. 4 | Diff. 5 |
|-----------|---------|---------|---------|---------|---------|
| Grille | 9x9 | 10x10 | 11x11 | 12x12 | 13x13 |
| Cles | 1 | 2 | 3 | 4 | 4 |
| Pieges | 3 | 7 | 9 | 11 | 13 |
| Ennemis | 1 | 2 | 3 | 4 | 4 |

---

## Entrainement RL (curriculum)

L'entrainement par curriculum progresse automatiquement a travers 6 etapes lorsque le taux de reussite depasse 60% sur les 200 derniers episodes :

| Etape | Grille | Cles | Pieges | Ennemis | Description |
|-------|--------|------|--------|---------|-------------|
| 0 | 7x7 | 0 | 0 | 0 | Navigation simple vers l'objectif |
| 1 | 7x7 | 0 | 1 | 0 | Introduction des pieges |
| 2 | 7x7 | 0 | 2 | 0 | Plus de pieges |
| 3 | 7x7 | 1 | 2 | 0 | Introduction cle + porte |
| 4 | 9x9 | 1 | 3 | 1 | Grille complete (difficulte 1) |
| 5 | 9x9 | 2 | 5 | 2 | Grille difficile (difficulte 2) |

Les instantanes sont sauvegardes periodiquement dans `models/snapshots/`, permettant de comparer l'agent a differents stades de son entrainement via `--rl-snapshot`.

---

## Benchmarks

Le systeme de benchmark evalue chaque solveur sur les memes niveaux generes proceduralement et produit un tableau de bord comprenant :

- **Taux de succes** : pourcentage de niveaux resolus
- **Efficacite** : nombre moyen de pas
- **Vitesse** : temps de calcul moyen
- **Distribution des recompenses** : diagramme en boite
- **Distribution des pas** : diagramme en boite
- **Comparaison normalisee** : vue d'ensemble multi-criteres

---

## Controles de la visualisation (mode Demo)

| Touche | Action |
|--------|--------|
| **Espace** | Pause / reprendre |
| **Fleche droite** | Avancer d'un pas |
| **Fleche haut** | Accelerer (min 50ms) |
| **Fleche bas** | Ralentir (max 1000ms) |
| **R** | Recommencer |
| **Q / Esc** | Quitter |

---

## Correspondance avec le cours

| Composante du projet | Semaine(s) du cours |
|---------------------|---------------------|
| BFS, A*, A* Safe | Semaine 5 -- Heuristiques, strategies de recherche, planification |
| DQN, MDP, recompenses | Semaine 5 -- MDP et apprentissage par renforcement |
| Reseau de neurones CNN (DQN) | Semaine 7 -- Reseaux de neurones et apprentissage profond (PyTorch) |
| Solveur LLM (Claude/GPT) | Semaine 9 -- TAL, Transformers, LLMs |
| Architecture d'agents, interface commune | Semaine 10 -- Agents intelligents |
| Benchmark, deploiement CLI | Semaine 13 -- IA operationnelle |

---

## Blog du projet

Un article de blog detaille expliquant les choix techniques, l'implementation et les resultats du projet est disponible dans [`docs/blog.md`](docs/blog.md).

---

## Contribuer

1. Forker le depot
2. Creer une branche (`git checkout -b feature/ma-fonctionnalite`)
3. Commiter les changements (`git commit -m "Ajout de ma fonctionnalite"`)
4. Pousser la branche (`git push origin feature/ma-fonctionnalite`)
5. Ouvrir une Pull Request

Merci de respecter le style de code existant et d'ajouter des tests si applicable.

---

## Licence

Ce projet est developpe dans un cadre academique (UQAM INF8790). Contactez les auteurs pour toute reutilisation.

---

## Credits

- **Generation de niveaux** : Algorithme de Prim randomise
- **Visualisation** : Pygame
- **RL** : PyTorch (DQN avec CNN)
- **LLM** : API Anthropic (Claude) / API OpenAI (ChatGPT)
- **Benchmarks** : Matplotlib
- **Cours** : INF8790 Fondements de l'Intelligence Artificielle -- UQAM 2025-2026

Projet realise dans le cadre du cours INF8790 -- Fondements de l'Intelligence Artificielle / UQAM 2025-2026.



!python training/train_traps.py --episodes 50000 --snapshot-every 2000
python main.py demo --solvers astar-safe rl --rl-snapshot models/dqn_traps_grid10.pt --grid-size 10 --simple
