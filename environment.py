# this file is discover-paths-rl/environment.py
# this file contains the PDWorld class which manages the grid environment

import copy
from constants import *

class PDWorld:
    """
    Manages the state of the 5x5 grid, including pickup/dropoff locations
    and the number of blocks. Enforces environment rules.
    """
    def __init__(self, pickup_locs=None, dropoff_locs=None):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        
        self.initial_pickup_locs = copy.deepcopy(pickup_locs) if pickup_locs else copy.deepcopy(DEFAULT_PICKUP_LOCS)
        self.initial_dropoff_locs = copy.deepcopy(dropoff_locs) if dropoff_locs else copy.deepcopy(DEFAULT_DROPOFF_LOCS)
        
        self.total_blocks_at_start = sum(self.initial_pickup_locs.values())
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.pickup_locs = copy.deepcopy(self.initial_pickup_locs)
        self.dropoff_locs = copy.deepcopy(self.initial_dropoff_locs)
        self.total_blocks_delivered = 0

    def is_terminal_state(self):
        """Checks if all blocks have been delivered."""
        return self.total_blocks_delivered == self.total_blocks_at_start

    def change_pickup_locations(self, new_pickup_locs):
        """
        For Experiment 4. Changes the pickup locations mid-run.
        The world state is updated immediately.
        """
        self.initial_pickup_locs = copy.deepcopy(new_pickup_locs)
        self.pickup_locs = copy.deepcopy(new_pickup_locs)
        # Update total blocks for terminal state check
        self.total_blocks_at_start = sum(self.initial_pickup_locs.values())
        # We must also reset delivered count to 0 for the *new* task
        self.total_blocks_delivered = 0 
        print(f"\n--- WORLD CHANGE: Pickup locations changed to: {new_pickup_locs} ---")

    def get_possible_actions(self, agent, other_agent):
        """Returns a list of all valid actions for the agent."""
        possible = []
        agent_pos = (agent.x, agent.y)
        other_pos = (other_agent.x, other_agent.y)
        
        for action, (dx, dy) in {'North': (0, -1), 'South': (0, 1), 'East': (1, 0), 'West': (-1, 0)}.items():
            next_x, next_y = agent.x + dx, agent.y + dy
            if 0 <= next_x < self.width and 0 <= next_y < self.height:
                if (next_x, next_y) != other_pos:
                    possible.append(action)
                    
        if not agent.has_block and agent_pos in self.pickup_locs and self.pickup_locs[agent_pos] > 0:
            possible.append('Pickup')
        # Assuming dropoff locs have infinite capacity as per prompt (or e.g., 5)
        if agent.has_block and agent_pos in self.dropoff_locs:
            possible.append('Dropoff')
        return possible

    def apply_action(self, agent, other_agent, action):
        """
        Applies an agent's action to the world.
        MODIFIES the agent object and the world state.
        Returns the reward.
        """
        agent_pos = (agent.x, agent.y)
        other_pos = (other_agent.x, other_agent.y)
        reward = -1  # Default cost for a move
        is_valid = True
        new_x, new_y = agent.x, agent.y

        if action == 'North': new_y -= 1
        elif action == 'South': new_y += 1
        elif action == 'East': new_x += 1
        elif action == 'West': new_x -= 1
        elif action == 'Pickup':
            if not agent.has_block and agent_pos in self.pickup_locs and self.pickup_locs[agent_pos] > 0:
                agent.has_block = True
                self.pickup_locs[agent_pos] -= 1
                reward = 13
            else:
                is_valid = False
                reward = -10
        elif action == 'Dropoff':
            if agent.has_block and agent_pos in self.dropoff_locs:
                agent.has_block = False
                self.dropoff_locs[agent_pos] += 1
                self.total_blocks_delivered += 1
                reward = 13
            else:
                is_valid = False
                reward = -10

        if action in ['North', 'South', 'East', 'West']:
            # Boundary check
            if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                is_valid = False
                reward = -10
                new_x, new_y = agent.x, agent.y
            # Blockage check
            elif (new_x, new_y) == other_pos:
                is_valid = False
                reward = -10
                new_x, new_y = agent.x, agent.y
        
        # Apply valid move
        if action in ['North', 'South', 'East', 'West'] and is_valid:
            agent.x, agent.y = new_x, new_y

        return reward