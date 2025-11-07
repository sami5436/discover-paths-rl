# this is discover-paths-rl/constants.py
# this file contains all constants used across the project

# --- CONSTANTS ---

# World Configuration
GRID_WIDTH = 5
GRID_HEIGHT = 5

# Note: Using (x, y) coordinates
DEFAULT_PICKUP_LOCS = {(1, 2): 5, (3, 3): 5}
DEFAULT_DROPOFF_LOCS = {(0, 0): 0, (0, 4): 0, (2, 2): 0, (4, 3): 0}

# Agent Start Positions
AGENT_F_START = {'x': 0, 'y': 2, 'has_block': False}
AGENT_M_START = {'x': 4, 'y': 2, 'has_block': False}

# RL Actions
ACTIONS = ['North', 'South', 'East', 'West', 'Pickup', 'Dropoff']