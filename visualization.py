import os
import matplotlib.pyplot as plt
import numpy as np
import random
from constants import GRID_WIDTH, GRID_HEIGHT, DEFAULT_PICKUP_LOCS, DEFAULT_DROPOFF_LOCS

class Visualization:
    """Groups all plotting and printing functions as static methods."""

    @staticmethod
    def plot_performance(steps_per_run_list, policy_switch_run, title):
        """Creates a plot showing steps per run."""
        plt.figure(figsize=(12, 6))
        plt.plot(steps_per_run_list, marker='o', linestyle='-', markersize=4)
        plt.title(f'{title}: Agent Performance Over Time (Steps per Terminal State)')
        plt.xlabel('Terminal State (Run Number)')
        plt.ylabel('Steps Taken to Complete')
        if policy_switch_run != -1:
            plt.axvline(x=policy_switch_run, color='r', linestyle='--', label=f'Policy Switch')
            plt.legend()
        
        plt.grid(True)
        
        # Save to a 'results' folder
        os.makedirs('results', exist_ok=True)
        filename = os.path.join('results', f'{title}_performance_plot.png')
        plt.savefig(filename)
        print(f"\nSaved performance plot to '{filename}'")
        plt.close() # Close the plot to save memory

    @staticmethod
    def plot_attractive_paths(q_table, agent_has_block, other_agent_pos, title):
        """
        Creates a "quiver plot" (arrow plot) to visualize the attractive paths.
        This is a "visually appealing" component for extra credit.
        """
        # Create a 5x5 grid for the path arrows
        # u = x-component of arrow, v = y-component of arrow
        u = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        v = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        
        # Create a 5x5 grid for the background colors
        grid_colors = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        
        movement_actions = ['North', 'South', 'East', 'West']
        action_vectors = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # This is the state for this specific grid cell
                # We fix the other agent's position for this visualization
                state = (x, y, agent_has_block, other_agent_pos[0], other_agent_pos[1])
                
                # Find the best *movement* action from this state
                best_move = ''
                max_q = -float('inf')
                
                for action in movement_actions:
                    q_val = q_table.get(state, {}).get(action, 0.0)
                    if q_val > max_q:
                        max_q = q_val
                        best_move = action
                
                # Store the vector for the best move
                if best_move in action_vectors:
                    dx, dy = action_vectors[best_move]
                    u[y, x] = dx
                    v[y, x] = dy * -1 # Y-axis is inverted in plots (0 is at top)
                
                # Set background colors
                if (x, y) in DEFAULT_PICKUP_LOCS:
                    grid_colors[y, x] = 1 # Blue
                elif (x, y) in DEFAULT_DROPOFF_LOCS:
                    grid_colors[y, x] = 2 # Green

        # --- Create the plot ---
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a colormap: 0=White, 1=Blue, 2=Green
        cmap = plt.get_cmap('Pastel2', 3)
        ax.imshow(grid_colors, cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
        
        # Draw the quiver plot (arrows)
        # We need to specify the center of each grid cell
        x_coords = np.arange(GRID_WIDTH)
        y_coords = np.arange(GRID_HEIGHT)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        ax.quiver(X, Y, u, v, color='black', scale=21, headwidth=4, headlength=5)

        # Configure the plot
        ax.set_title(title, fontsize=14)
        ax.set_xticks(np.arange(GRID_WIDTH))
        ax.set_yticks(np.arange(GRID_HEIGHT))
        ax.set_xticklabels(np.arange(GRID_WIDTH))
        ax.set_yticklabels(np.arange(GRID_HEIGHT))
        ax.set_xticks(np.arange(-.5, GRID_WIDTH, 1), minor=True)
        ax.set_yticks(np.arange(-.5, GRID_HEIGHT, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.set_aspect('equal') # Make squares square
        
        # Add a legend for colors
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=cmap(0), label='Empty'),
                           plt.Rectangle((0, 0), 1, 1, color=cmap(1), label='Pickup (P)'),
                           plt.Rectangle((0, 0), 1, 1, color=cmap(2), label='Dropoff (D)')]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.gca().invert_yaxis() # Put (0,0) at the top-left

        # Save to a 'results' folder
        os.makedirs('results', exist_ok=True)
        filename = os.path.join('results', f'{title}_path_plot.png')
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved path visualization to '{filename}'")
        plt.close()

    @staticmethod
    def print_q_table_sample(q_table, num_states=5):
        """Prints a small, random sample of the Q-table."""
        print(f"\n--- Q-Table Sample (Size: {len(q_table)} states) ---")
        if not q_table:
            print("Q-Table is empty.")
            return
        
        # Get a random sample of states
        states_to_print = random.sample(list(q_table.keys()), min(num_states, len(q_table)))
        
        for state in states_to_print:
            actions = q_table[state]
            # Format actions for printing
            actions_str = ", ".join(f"{act}: {val:.2f}" for act, val in actions.items())
            print(f"State: {state}\n  Actions: {actions_str}")