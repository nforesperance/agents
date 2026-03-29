# AI Puzzle Solver Showdown : trois paradigmes d'IA face a face

*Projet de session -- INF8790 Fondements de l'Intelligence Artificielle, UQAM 2025-2026*

---

## Introduction

Comment comparer equitablement des approches d'intelligence artificielle fondamentalement differentes ? C'est la question centrale de ce projet. Plutot que d'etudier chaque paradigme isolement, nous avons concu un banc d'essai unifie ou **recherche classique** (A\*, BFS), **apprentissage par renforcement** (DQN) et **grands modeles de langage** (Claude, ChatGPT) s'affrontent sur la meme tache : resoudre des puzzles generes proceduralement.

L'idee est simple mais reveladora : placer ces trois familles d'algorithmes devant le meme probleme permet de mettre en evidence leurs forces, leurs faiblesses et leurs compromis respectifs. Un algorithme A\* trouve le chemin optimal en quelques millisecondes, mais ignore les dangers et meurt sur les pieges. Un agent DQN apprend par essai-erreur a naviguer prudemment, mais necessite des heures d'entrainement. Un LLM peut raisonner sur la situation en langage naturel, mais chaque pas coute un appel API et prend des secondes.

Ce projet couvre trois grands axes du cours INF8790 :
- **Semaine 5** : Algorithmes de recherche (BFS, A\*), processus de decision markoviens (MDP) et apprentissage par renforcement.
- **Semaine 7** : Reseaux de neurones et apprentissage profond, ici appliques via un CNN dans l'architecture DQN.
- **Semaine 9** : Traitement automatique du langage naturel et grands modeles de langage (Transformers, LLMs).

---

## Le jeu de puzzle

### Conception du moteur

Le moteur de jeu (`game/engine.py`) implemente un environnement de type grille avec des mecaniques inspirees des roguelikes classiques. L'etat du jeu est encapsule dans une dataclass `GameState` qui contient la grille, la position du joueur, les cles collectees, les points de vie, les positions des ennemis et le compteur de pas.

Le coeur du moteur est la methode `step(action)`, qui applique une action et retourne le nouvel etat, la recompense et un indicateur de fin de partie :

```python
def step(self, action: int) -> tuple[GameState, float, bool]:
    """Apply action, return (new_state, reward, done).

    Reward scheme (dense, to help RL learn):
      +100  reach goal
       +10  pick up a key
        -1  each step (encourages efficiency)
       -10  hit a trap
       -20  hit an enemy
       -50  timeout
    """
```

Ce schema de recompenses dense est crucial pour l'apprentissage par renforcement : une recompense sparse (seulement +100 pour le goal) rendrait l'exploration quasi impossible pour l'agent DQN sur des grilles de taille 9x9 ou plus.

Le moteur fournit deux representations de l'etat :
- **`to_observation()`** : un tenseur multi-canal (5 x H x W) pour l'agent RL, ou chaque canal encode un aspect de l'etat (tuiles normalisees, position du joueur, ennemis, ratio de cles, sante).
- **`to_text()`** : une representation textuelle de la grille avec legende, destinee au solveur LLM.

Cette dualite de representation illustre un point fondamental : la meme information peut etre encodee de manieres radicalement differentes selon le consommateur. Le CNN du DQN travaille sur des matrices numeriques ; le LLM travaille sur du texte structure.

### Generation procedurale avec l'algorithme de Prim

Les niveaux sont generes par `game/level_generator.py` en utilisant une variante randomisee de l'algorithme de Prim pour la generation de labyrinthes. L'algorithme fonctionne comme suit :

```python
def _generate_maze(self, rows: int, cols: int) -> np.ndarray:
    grid = np.full((rows, cols), WALL, dtype=np.int32)
    sr, sc = 1, 1
    grid[sr, sc] = FLOOR
    frontiers = []

    # Ajouter les frontieres de la cellule initiale
    add_frontiers(sr, sc)

    while frontiers:
        # Choisir une frontiere aleatoire
        idx = self.rng.randrange(len(frontiers))
        fr, fc = frontiers.pop(idx)

        if grid[fr, fc] != FLOOR:
            # Trouver les voisins deja creuses
            neighbors = [...]
            if neighbors:
                # Connecter au labyrinthe
                nr, nc, wr, wc = self.rng.choice(neighbors)
                grid[fr, fc] = FLOOR
                grid[wr, wc] = FLOOR
                add_frontiers(fr, fc)
```

Le principe est le suivant : on part d'une grille entierement remplie de murs, on creuse une cellule de depart, puis on maintient une liste de « frontieres » (cellules a deux pas de distance d'une cellule deja creusee). A chaque iteration, on choisit une frontiere au hasard, on la creuse en connectant le mur intermediaire, et on ajoute ses propres frontieres a la liste. Cela produit un labyrinthe parfait (un seul chemin entre deux points quelconques).

Pour rendre les niveaux plus interessants et offrir plusieurs chemins possibles, le generateur ouvre ensuite des passages supplementaires de maniere controlee :

```python
extra = (rows * cols) // 8
for _ in range(extra):
    r = self.rng.randint(1, rows - 2)
    c = self.rng.randint(1, cols - 2)
    if grid[r, c] == WALL:
        adj_floor = sum(1 for ... if grid[r+dr, c+dc] == FLOOR)
        if 1 <= adj_floor <= 2:
            grid[r, c] = FLOOR
```

La contrainte `1 <= adj_floor <= 2` empeche la creation de larges zones ouvertes tout en creant des chemins alternatifs. Cela donne aux solveurs plusieurs options de routes, rendant la comparaison plus riche.

Enfin, chaque niveau genere est verifie par un BFS pour garantir qu'il est solvable (l'objectif est atteignable depuis le depart). Si ce n'est pas le cas, le generateur regenere un niveau.

---

## Les solveurs classiques : BFS et A*

### BFS (Breadth-First Search)

Le solveur BFS explore l'espace d'etats couche par couche. L'etat de recherche est un triplet `(row, col, frozenset_of_keys)` qui encode la position du joueur et l'ensemble des cles collectees. Cela est necessaire car collecter une cle change fondamentalement l'espace d'etats accessible (les portes s'ouvrent).

BFS garantit le chemin le plus court en nombre de pas. Cependant, il ne tient aucun compte des dangers : il peut router le joueur a travers plusieurs pieges consecutifs, causant la mort par perte de HP.

### A* avec heuristique Manhattan

A\* ameliore BFS en utilisant une heuristique admissible pour guider la recherche vers l'objectif. L'heuristique utilisee est la distance de Manhattan vers la cle non collectee la plus proche, plus la distance de cette cle vers l'objectif :

```python
def heuristic(s: SearchState) -> float:
    r, c, keys = s
    if len(keys) < total_keys:
        uncollected = [kp for i, kp in enumerate(key_positions) if i not in keys]
        if uncollected:
            nearest = min(uncollected, key=lambda kp: abs(kp[0]-r) + abs(kp[1]-c))
            return (abs(nearest[0]-r) + abs(nearest[1]-c)
                    + abs(goal_pos[0]-nearest[0]) + abs(goal_pos[1]-nearest[1]))
    return abs(goal_pos[0] - r) + abs(goal_pos[1] - c)
```

Cette heuristique est admissible (elle ne surestime jamais le cout reel) car la distance de Manhattan est un minorant du nombre de pas dans une grille avec obstacles. Le resultat est un chemin de longueur optimale en nombre de pas, identique a BFS, mais explore en general beaucoup moins de noeuds.

### A* Safe : conscience des dangers

La variante A\* Safe illustre un concept fondamental : la modification de la fonction de cout change radicalement le comportement de la recherche. Au lieu d'un cout uniforme de 1 par pas, A\* Safe penalise les tuiles dangereuses :

```python
def tile_cost(r: int, c: int) -> float:
    cost = 1.0
    if tile == TRAP:
        cost += 10.0    # fortement decourager les pieges
    if (r, c) in enemy_cells:
        cost += 5.0     # decourager les cases adjacentes aux ennemis
    return cost
```

Les zones de danger des ennemis sont precalculees : chaque ennemi rend dangereuses les 5 cellules (sa position + 4 voisins cardinaux). Le resultat est un chemin souvent plus long en nombre de pas, mais beaucoup plus sur. Sur les cartes a haute difficulte, A\* Safe a un taux de survie significativement superieur a A\* classique.

---

## L'agent DQN : apprentissage par renforcement profond

### Architecture du reseau

L'agent DQN (`solvers/rl_solver.py`) utilise un reseau de neurones convolutionnel pour approximer la fonction Q(s, a). L'architecture est concue pour traiter des observations sous forme de grille :

```python
class DQN(nn.Module):
    def __init__(self, in_channels=5, grid_size=9, n_actions=5):
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_size = 64 * grid_size * grid_size
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
```

Le choix du CNN est motive par la nature spatiale de l'observation : les relations de voisinage entre les tuiles sont essentielles pour la prise de decision. Les 5 canaux d'entree encodent respectivement le type de tuile (normalise entre 0 et 1), la position du joueur (one-hot), les positions des ennemis (one-hot), le ratio de cles collectees (scalaire diffuse) et la sante (scalaire diffuse).

La sortie est un vecteur de 5 valeurs Q, une par action possible (UP, DOWN, LEFT, RIGHT, WAIT). L'action WAIT est importante car elle permet a l'agent d'attendre qu'un ennemi se deplace avant de traverser un couloir.

### Mecanismes d'entrainement

L'entrainement DQN repose sur trois mecanismes cles :

1. **Experience Replay** : Un buffer circulaire de 50 000 transitions `(s, a, r, s', done)` stocke les experiences passees. A chaque pas d'entrainement, un mini-lot de 64 transitions est echantillonne aleatoirement. Cela brise les correlations temporelles entre les experiences consecutives, stabilisant l'apprentissage.

2. **Target Network** : Un second reseau (copie du reseau principal) est utilise pour calculer les valeurs Q cibles. Il est mis a jour tous les 10 episodes par copie des poids du reseau principal. Cela evite les oscillations causees par un objectif qui change a chaque mise a jour.

3. **Epsilon-greedy** : La politique d'exploration commence avec epsilon = 1.0 (exploration totale) et decroit progressivement vers 0.05 avec un facteur de 0.995 par episode. Cela permet a l'agent d'explorer largement au debut, puis de se concentrer sur l'exploitation des connaissances acquises.

La fonction de perte est le Smooth L1 (Huber Loss), plus robuste aux valeurs aberrantes que le MSE, avec un gradient clipping a 1.0 pour eviter les explosions de gradient :

```python
loss = nn.functional.smooth_l1_loss(q_values, target)
self.optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
self.optimizer.step()
```

### Apprentissage par curriculum

L'un des defis majeurs de l'entrainement RL sur notre puzzle est la complexite combinatoire : sur une grille 9x9 avec cles, pieges et ennemis, un agent aleatoire a une probabilite infime de trouver l'objectif. Sans succes initial, il n'y a pas de signal de recompense positif, et l'agent n'apprend rien.

La solution adoptee est l'**apprentissage par curriculum** (`training/train_rl.py`), inspire des travaux de Bengio et al. (2009). L'agent progresse a travers 6 etapes de difficulte croissante :

| Etape | Configuration | Objectif pedagogique |
|-------|--------------|---------------------|
| 0 | 7x7, aucun obstacle | Apprendre a naviguer vers l'objectif |
| 1 | 7x7, 1 piege | Apprendre a eviter les pieges |
| 2 | 7x7, 2 pieges | Renforcer l'evitement de dangers |
| 3 | 7x7, 1 cle + porte, 2 pieges | Apprendre la sequence cle-porte-objectif |
| 4 | 9x9, 1 cle, 3 pieges, 1 ennemi | Environnement complet simple |
| 5 | 9x9, 2 cles, 5 pieges, 2 ennemis | Environnement complet difficile |

L'avancement est automatique : lorsque le taux de reussite depasse 60% sur les 200 derniers episodes, l'agent passe a l'etape suivante. Ce seuil a ete choisi empiriquement comme un bon compromis entre maitrise et progression.

### Systeme d'instantanes

Le systeme d'instantanes (snapshots) sauvegarde periodiquement les poids du modele pendant l'entrainement (par defaut tous les 500 episodes). Cela remplit deux fonctions :

1. **Analyse du progres** : On peut charger n'importe quel instantane via `--rl-snapshot` et observer le comportement de l'agent a differents stades de son entrainement. Par exemple, comparer l'agent a 500 episodes (debut du curriculum) et a 5000 episodes (fin du curriculum).

2. **Robustesse** : Si l'entrainement diverge ou si l'on souhaite reprendre a un point anterieur, les instantanes servent de points de restauration.

Les instantanes sont sauvegardes dans `models/snapshots/` avec un nom incluant la taille de grille, la difficulte et le numero d'episode, par exemple `dqn_grid9_d1_ep2500.pt`.

---

## Le solveur LLM : raisonnement en langage naturel

### Conception du prompt

Le solveur LLM (`solvers/llm_solver.py`) represente une approche radicalement differente : au lieu de chercher dans un graphe ou d'apprendre une politique, il demande a un modele de langage de raisonner sur la situation.

Le prompt systeme definit les regles du jeu et les strategies recommandees :

```python
SYSTEM_PROMPT = """\
You are an AI agent solving a grid-based puzzle game.
You see the grid and must choose actions to reach the goal (G).

Rules:
- P = your position. Move with: UP, DOWN, LEFT, RIGHT, WAIT
- # = walls (impassable)
- . = floor (safe)
- X = trap (damages you, avoid if possible)
- K = key (collect all keys before you can open doors)
- D = door (requires all keys collected to pass through)
- G = goal (reach this to win)
- E = enemy (moves around, damages you on contact, avoid)

Respond with ONLY a JSON object: {"action": "UP"} ...
"""
```

A chaque pas, l'etat courant est envoye sous forme textuelle, et le LLM repond avec un objet JSON contenant l'action choisie. Un historique de conversation (limite aux 10 derniers echanges) fournit le contexte des actions precedentes.

### Support multi-fournisseur

Le solveur prend en charge deux fournisseurs de LLM :
- **Claude** (API Anthropic) : utilise par defaut.
- **ChatGPT** (API OpenAI) : selectionnable via `--llm-provider openai`.

L'abstraction est realisee dans la methode `_ask()`, qui encapsule les differences d'API entre les deux fournisseurs. Le parsing de la reponse est robuste : il cherche d'abord un objet JSON dans la reponse, puis, en cas d'echec, recherche les mots-cles d'action dans le texte.

### Forces et limites

Le solveur LLM presente des caracteristiques uniques :
- **Raisonnement explicite** : Le LLM peut potentiellement planifier plusieurs coups a l'avance et comprendre des concepts abstraits comme "collecter les cles avant d'aller a la porte".
- **Zero-shot** : Aucun entrainement specifique n'est necessaire. Le LLM resout le puzzle a partir de sa comprehension generale du monde.
- **Cout et latence** : Chaque pas necessite un appel API, ce qui rend le solveur beaucoup plus lent (de l'ordre de la seconde par pas) et couteux (facturation par token).
- **Fiabilite du format** : Le LLM peut parfois generer des reponses mal formatees, necessitant un parsing robuste avec fallback.

---

## Le cadre de benchmark

### Evaluateur multi-niveaux

Le module `benchmarks/runner.py` offre une evaluation rigoureuse et reproductible. Le principe est simple : generer N niveaux avec une graine fixe, executer chaque solveur sur exactement les memes niveaux, et collecter les metriques.

```python
def benchmark(solvers, num_levels=20, grid_size=9, difficulty=1, seed=42):
    gen = LevelGenerator(seed=seed)
    levels = gen.generate_batch(num_levels, size=grid_size, difficulty=difficulty)

    for solver in solvers:
        for level in levels:
            solver.reset()
            metrics = run_solver_on_level(solver, level)
            # Collecter: solved, steps, reward, time_ms, health
```

Les metriques collectees pour chaque paire (solveur, niveau) sont :
- **Resolu** : le joueur a-t-il atteint l'objectif ?
- **Pas** : nombre de pas effectues
- **Recompense** : recompense cumulative
- **Temps** : duree de calcul en millisecondes
- **Sante** : HP restants a la fin

### Tableau de bord

Le module `ui/dashboard.py` produit un tableau de bord en 6 panneaux via Matplotlib :
1. Taux de succes (diagramme a barres)
2. Nombre moyen de pas (diagramme a barres)
3. Temps de calcul moyen (diagramme a barres)
4. Distribution des recompenses (diagramme en boite)
5. Distribution des pas (diagramme en boite)
6. Comparaison normalisee multi-criteres

Ce tableau de bord peut etre sauvegarde en image via `--save-plot results.png` pour inclusion dans les rapports.

---

## Resultats et analyse

### Solveurs classiques : rapidite sans discernement

Les solveurs A\* et BFS trouvent systematiquement un chemin vers l'objectif en quelques millisecondes. Leur taux de resolution est de 100% au sens de la planification (un chemin existe toujours par construction du generateur). Cependant, en termes de survie, leur taux chute significativement sur les niveaux de haute difficulte : en ignorant les pieges, un chemin traversant 4 pieges consomme 100 HP, tuant le joueur avant d'atteindre l'objectif.

A\* Safe corrige cette faiblesse en penalisant les tuiles dangereuses. Le compromis est clair dans les metriques : les chemins sont plus longs en nombre de pas (et donc une recompense brute plus basse a cause du cout de -1 par pas), mais le taux de survie est beaucoup plus eleve. Sur des niveaux de difficulte 3 et plus, A\* Safe surpasse nettement A\* et BFS en recompense finale, car eviter un piege (-10 de recompense + -25 HP) vaut largement quelques pas supplementaires (-1 chacun).

### Agent DQN : apprentissage progressif

Le comportement de l'agent DQN varie enormement selon son stade d'entrainement. Un agent non entraine (ou entraine seulement sur les premieres etapes du curriculum) agit quasi aleatoirement et echoue systematiquement. Apres un entrainement complet (5000-10000 episodes avec curriculum), l'agent apprend a :
- Naviguer efficacement vers les cles et l'objectif
- Eviter les pieges dans la plupart des cas
- Attendre parfois avant de traverser un couloir avec ennemi

Neanmoins, l'agent DQN generalise difficilement a des configurations qu'il n'a jamais vues pendant l'entrainement. Un agent entraine sur des grilles 9x9 performe mal sur des grilles 11x11, car la taille de l'observation change (le padding est utilise mais les motifs spatiaux different). Le curriculum aide a attenuer ce probleme en exposant l'agent a des tailles variees (7x7 puis 9x9).

### Solveur LLM : raisonnement couteux

Le LLM demontre une capacite de raisonnement remarquable sur les niveaux simples : il comprend la necessite de collecter les cles avant d'ouvrir la porte, evite les pieges visibles, et planifie des itineraires coherents. Cependant, ses performances se degradent sur les niveaux complexes, en partie parce que la representation textuelle de la grille devient difficile a analyser spatialement pour un modele de langage.

Le facteur limitant principal est la latence : chaque pas prend entre 500ms et 2s selon le fournisseur et le modele. Sur un niveau de 50 pas, le LLM met pres d'une minute la ou A\* prend 2 millisecondes. En benchmark sur 20 niveaux, le solveur LLM est facilement 1000 fois plus lent que les solveurs classiques.

### Synthese comparative

| Critere | A\*/BFS | A\* Safe | DQN (entraine) | LLM |
|---------|---------|----------|-----------------|-----|
| Vitesse | Tres rapide (~ms) | Tres rapide (~ms) | Rapide (~ms) | Tres lent (~s) |
| Taux de succes (facile) | Eleve | Tres eleve | Moyen-Eleve | Moyen |
| Taux de succes (difficile) | Moyen (mort sur pieges) | Eleve | Variable | Moyen-Faible |
| Generalisation | Parfaite | Parfaite | Limitee | Bonne en theorie |
| Entrainement requis | Aucun | Aucun | Plusieurs heures | Aucun |
| Cout par execution | Nul | Nul | Nul | Cout API |

---

## Conclusion et lecons apprises

Ce projet met en lumiere un principe fondamental de l'IA : **il n'existe pas de solveur universel**. Chaque paradigme excelle dans certaines conditions et echoue dans d'autres.

Les algorithmes de recherche classiques (Semaine 5) brillent par leur garantie d'optimalite, leur vitesse et leur determinisme. Ils sont le choix evident lorsque le probleme peut etre modelise comme un graphe d'etats et que la fonction de cout est bien definie. L'ajout d'une conscience des dangers (A\* Safe) montre comment une simple modification de la fonction de cout transforme radicalement le comportement.

L'apprentissage par renforcement profond (Semaines 5 et 7) offre la promesse d'apprendre des politiques complexes a partir de l'experience seule. Le curriculum learning est essentiel : sans lui, l'agent ne recoit jamais de signal positif et ne peut pas apprendre. Le DQN illustre concretement les concepts de MDP, fonction de valeur, et approximation par reseau de neurones.

Les grands modeles de langage (Semaine 9) representent un paradigme emergent ou le raisonnement se fait en langage naturel. Leur capacite zero-shot est impressionnante, mais leur cout et leur latence les rendent impraticables pour des taches necessitant des milliers de decisions rapides. Ils sont plus adaptes a la planification de haut niveau qu'a l'execution pas-a-pas.

En definitive, le systeme ideal serait probablement hybride : un LLM pour la planification strategique, un algorithme de recherche pour l'execution tactique, et un agent RL pour les situations dynamiques imprevues. C'est d'ailleurs la direction que prend la recherche actuelle en IA, comme en temoignent les travaux sur les agents LLM augmentes d'outils et de planificateurs.

Ce projet nous a egalement permis de mettre en pratique les concepts de la **Semaine 10** (agents intelligents) en concevant une interface commune (`BaseSolver`) et un cadre d'evaluation unifie, et ceux de la **Semaine 13** (IA operationnelle) en deployant le tout comme un outil CLI complet avec visualisation temps reel et benchmarks automatises.

---

*Projet realise dans le cadre du cours INF8790 -- Fondements de l'Intelligence Artificielle, UQAM 2025-2026.*
