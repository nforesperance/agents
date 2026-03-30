"""Global configuration for the AI Puzzle Solver Showdown."""

# Grid settings
TILE_SIZE = 40
GRID_MIN = 7
GRID_MAX = 15
DEFAULT_GRID_SIZE = 9

# Tile types
FLOOR = 0
WALL = 1
TRAP = 2
KEY = 3
DOOR = 4
GOAL = 5
ENEMY = 6
START = 7

TILE_NAMES = {
    FLOOR: "floor",
    WALL: "wall",
    TRAP: "trap",
    KEY: "key",
    DOOR: "door",
    GOAL: "goal",
    ENEMY: "enemy",
    START: "start",
}

# Colors (RGB)
COLORS = {
    FLOOR: (40, 40, 60),
    WALL: (100, 100, 120),
    TRAP: (200, 60, 60),
    KEY: (255, 215, 0),
    DOOR: (139, 90, 43),
    GOAL: (0, 255, 100),
    ENEMY: (200, 0, 200),
    START: (80, 140, 255),
    "player": (0, 180, 255),
    "bg": (20, 20, 30),
    "text": (220, 220, 220),
    "panel": (30, 30, 50),
    "highlight": (255, 255, 100),
    "classical": (0, 200, 255),
    "rl": (255, 100, 50),
    "llm": (100, 255, 100),
}

# Key colors for multi-key puzzles
KEY_COLORS = [
    (255, 215, 0),    # gold
    (0, 200, 255),    # cyan
    (255, 100, 100),  # red
    (100, 255, 100),  # green
]

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
WAIT = 4

ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT", WAIT: "WAIT"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), WAIT: (0, 0)}

# RL settings
RL_LEARNING_RATE = 1e-3
RL_GAMMA = 0.99
RL_EPSILON_START = 1.0
RL_EPSILON_END = 0.05
RL_EPSILON_DECAY = 0.995
RL_BATCH_SIZE = 64
RL_MEMORY_SIZE = 50000
RL_TARGET_UPDATE = 10
RL_MAX_STEPS = 200

# LLM settings
LLM_MAX_STEPS = 100

# Visualization
FPS = 10
SOLVER_STEP_DELAY = 1000  # ms between steps in visualization
WINDOW_WIDTH = 1900 #1400
WINDOW_HEIGHT = 1040 #800
