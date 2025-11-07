# this file is discover-paths-rl/agent.py
# it contains the Agent and RLAgentController classes 
# which implement the agents and their RL logic

import random
from constants import ACTIONS

class Agent:
    """A simple class to hold the state of an agent."""
    def __init__(self, name, x, y, has_block=False):
        self.name = name
        self.x = x
        self.y = y
        self.has_block = has_block # True if carrying a block, False otherwise

    def reset(self, x, y, has_block=False): 
        """Resets the agent's state."""
        self.x = x
        self.y = y
        self.has_block = has_block

    def __str__(self):
        return f"Agent {self.name} at ({self.x}, {self.y}), holding: {self.has_block}"

class RLAgentController:
    """
    The "brain" for an agent. Owns the Q-table and all RL logic.
    Implements Option (a): separate Q-tables, but state includes other agent.
    """
    def __init__(self, agent, other_agent, world, learning_rate, discount_factor):
        self.agent = agent
        self.other_agent = other_agent
        self.world = world
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # Key: state tuple, Value: {action: q_value}

    def get_current_state(self): # the current state from this agent's perspective
        """Generates the state tuple from the agent's perspective."""
        return (self.agent.x, self.agent.y, self.agent.has_block, 
                self.other_agent.x, self.other_agent.y)

    def get_q_value(self, state, action):
        """Helper to get Q-value, initializing if not present."""
        if state not in self.q_table: # if state not in Q-table, initialize
            self.q_table[state] = {act: 0.0 for act in ACTIONS} # all actions start at 0.0
        if action not in self.q_table[state]: # if action not in Q-table for this state, initialize
             self.q_table[state][action] = 0.0 # default Q-value
        return self.q_table[state][action] # return the Q-value if present (this will return 0.0 if just initialized)

    def get_max_q_action(self, state, possible_actions):
        """Finds the action with the highest Q-value from a list of possible actions."""
        if not possible_actions:
            return None, 0.0
        max_q = -float('inf')
        best_actions = []
        for action in possible_actions: # iterate only over possible actions
            q_val = self.get_q_value(state, action)
            if q_val > max_q: # found a new max
                max_q = q_val
                best_actions = [action]
            elif q_val == max_q: # found another action with same max Q-value
                best_actions.append(action)
        return random.choice(best_actions), max_q # break ties randomly

    def choose_action(self, policy, possible_actions):
        """Chooses an action based on the current policy."""
        state = self.get_current_state()
        
        if not possible_actions:
            return None
        # Policy rule: P/D always takes precedence
        if 'Pickup' in possible_actions:
            return 'Pickup'
        if 'Dropoff' in possible_actions:
            return 'Dropoff'

        if policy == 'PRANDOM':
            return random.choice(possible_actions)
        
        elif policy == 'PEXPLOIT':
            best_action, _ = self.get_max_q_action(state, possible_actions)
            if random.random() < 0.8: # 80% exploit
                return best_action
            else: # 20% explore
                exploration_choices = [a for a in possible_actions if a != best_action]
                if not exploration_choices:
                    return best_action
                return random.choice(exploration_choices)

        elif policy == 'PGREEDY':
            best_action, _ = self.get_max_q_action(state, possible_actions)
            return best_action
            
        raise ValueError(f"Unknown policy: {policy}")

    def update_q_table(self, old_state, action, reward, new_state, new_possible_actions):
        """Performs the Q-Learning update rule."""
        old_q = self.get_q_value(old_state, action)
        _ , max_next_q = self.get_max_q_action(new_state, new_possible_actions)
        temporal_difference = reward + (self.discount_factor * max_next_q) - old_q
        new_q = old_q + (self.learning_rate * temporal_difference)
        self.q_table[old_state][action] = new_q

    def update_sarsa_table(self, old_state, action, reward, new_state, next_action):
        """Performs the SARSA update rule."""
        old_q = self.get_q_value(old_state, action)
        next_q = self.get_q_value(new_state, next_action) if next_action else 0.0
        temporal_difference = reward + (self.discount_factor * next_q) - old_q
        new_q = old_q + (self.learning_rate * temporal_difference)
        self.q_table[old_state][action] = new_q