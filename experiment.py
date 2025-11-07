# this is discover-paths-rl/experiment.py
# this file contains the ExperimentRunner class
# which sets up and runs experiments based on configurations

from environment import PDWorld
from agent import Agent, RLAgentController
from visualization import Visualization
from constants import AGENT_F_START, AGENT_M_START

class ExperimentRunner:
    """Runs a single, complete experiment based on a configuration."""
    
    def __init__(self, config):
        self.config = config
        
        print("\n" + "="*50)
        print(f"Initializing Experiment: {self.config['name']}")
        print("="*50)
        
        # Init environment
        self.world = PDWorld()
        
        # Init agents
        self.agent_f = Agent('F', **AGENT_F_START)
        self.agent_m = Agent('M', **AGENT_M_START)
        
        # Init controllers
        self.controller_f = RLAgentController(
            self.agent_f, self.agent_m, self.world,
            self.config['learning_rate'], self.config['discount_factor']
        )
        self.controller_m = RLAgentController(
            self.agent_m, self.agent_f, self.world,
            self.config['learning_rate'], self.config['discount_factor']
        )
        
        self.agents = [self.agent_f, self.agent_m]
        self.controllers = [self.controller_f, self.controller_m]

        # Stats tracking
        self.steps_per_run = []
        self.all_manhattan_distances = []
        self.total_rewards = {self.agent_f.name: 0, self.agent_m.name: 0}
        self.terminal_states_reached = 0

    def run(self):
        """Runs the simulation loop for this experiment."""
        
        print(f"Starting simulation... {self.config['total_steps']} steps.")
        print(f"Algorithm: {self.config['algorithm']}, LR(α): {self.config['learning_rate']}")
        
        # Get policy schedule
        policy_schedule = self.config['policy_schedule']
        policy_index = 0
        current_policy = policy_schedule[policy_index][1]
        policy_switch_step = policy_schedule[policy_index][0]
        first_policy_switch_run = -1 # Run number where policy switched
        
        current_run_steps = 0
        
        # --- Handle different loop logic for Q-Learning vs SARSA ---
        if self.config['algorithm'] == 'Q_LEARNING':
            self._run_q_learning_loop(policy_schedule, policy_index, current_policy, policy_switch_step, first_policy_switch_run, current_run_steps)
        elif self.config['algorithm'] == 'SARSA':
            self._run_sarsa_loop(policy_schedule, policy_index, current_policy, policy_switch_step, first_policy_switch_run, current_run_steps)
        else:
            raise ValueError(f"Unknown algorithm: {self.config['algorithm']}")
            
        self._print_results()

    def _run_q_learning_loop(self, policy_schedule, policy_index, current_policy, policy_switch_step, first_policy_switch_run, current_run_steps):
        """Main simulation loop for Q-Learning."""
        
        for step in range(self.config['total_steps']):
            # --- Policy Switching Logic ---
            if policy_index < len(policy_schedule) - 1 and step >= policy_switch_step:
                if first_policy_switch_run == -1:
                    first_policy_switch_run = len(self.steps_per_run)
                policy_index += 1
                switch_step_limit, new_policy = policy_schedule[policy_index]
                current_policy = new_policy
                policy_switch_step += switch_step_limit
                print(f"--- Step {step}: Switching policy to {current_policy} ---")

            # --- Experiment 4: World Change Logic ---
            if 'Exp_4_Adaptability' in self.config['name'] and self.terminal_states_reached == 3:
                self.world.change_pickup_locations({(1, 2): 5, (4, 4): 5}) # New locs
                self.terminal_states_reached = 3.1 # Hack to prevent re-triggering

            # --- Agent Turn ---
            agent_turn = step % 2
            controller = self.controllers[agent_turn]
            
            # 1. Get State (S)
            old_state = controller.get_current_state()
            possible_actions = self.world.get_possible_actions(controller.agent, controller.other_agent)
            
            # 2. Choose Action (A)
            action = controller.choose_action(current_policy, possible_actions)
            if action is None:
                continue # Agent is trapped
            
            # 3. Apply Action, get Reward (R)
            reward = self.world.apply_action(controller.agent, controller.other_agent, action)
            self.total_rewards[controller.agent.name] += reward
            
            # 4. Get New State (S') and New Actions
            new_state = controller.get_current_state()
            new_possible_actions = self.world.get_possible_actions(controller.agent, controller.other_agent)
            
            # 5. Update Q-Table
            controller.update_q_table(old_state, action, reward, new_state, new_possible_actions)
            
            # --- Stats and Terminal State Check ---
            self.first_policy_switch_run = first_policy_switch_run # Store this for plotting
            current_run_steps = self._track_stats_and_reset(step, current_run_steps)
            current_run_steps += 1

    def _run_sarsa_loop(self, policy_schedule, policy_index, current_policy, policy_switch_step, first_policy_switch_run, current_run_steps):
        """Main simulation loop for SARSA."""
        
        # SARSA requires choosing the first action *before* the loop
        sarsa_s_a = [None, None] # To store (state, action) for F and M
        for i in [0, 1]:
            controller = self.controllers[i]
            state = controller.get_current_state()
            possible = self.world.get_possible_actions(controller.agent, controller.other_agent)
            action = controller.choose_action(current_policy, possible)
            sarsa_s_a[i] = {'state': state, 'action': action}

        for step in range(self.config['total_steps']):
            # --- Policy Switching Logic ---
            if policy_index < len(policy_schedule) - 1 and step >= policy_switch_step:
                if first_policy_switch_run == -1:
                    first_policy_switch_run = len(self.steps_per_run)
                policy_index += 1
                switch_step_limit, new_policy = policy_schedule[policy_index]
                current_policy = new_policy
                policy_switch_step += switch_step_limit
                print(f"--- Step {step}: Switching policy to {current_policy} ---")

            # --- Experiment 4: World Change Logic ---
            if 'Exp_4_Adaptability' in self.config['name'] and self.terminal_states_reached == 3:
                self.world.change_pickup_locations({(1, 2): 5, (4, 4): 5}) # New locs
                self.terminal_states_reached = 3.1 # Hack to prevent re-triggering

            # --- Agent Turn ---
            agent_turn = step % 2
            controller = self.controllers[agent_turn]

            # 1. Get OLD State and Action (S, A)
            old_state = sarsa_s_a[agent_turn]['state']
            action = sarsa_s_a[agent_turn]['action']
            
            if action is None:
                continue # Agent was trapped

            # 2. Apply Action, get Reward (R) and New State (S')
            reward = self.world.apply_action(controller.agent, controller.other_agent, action)
            self.total_rewards[controller.agent.name] += reward
            new_state = controller.get_current_state()

            # 3. Get New State's Actions and Choose NEXT Action (A')
            new_possible_actions = self.world.get_possible_actions(controller.agent, controller.other_agent)
            next_action = controller.choose_action(current_policy, new_possible_actions)

            # 4. Update Q-Table (using S, A, R, S', A')
            controller.update_sarsa_table(old_state, action, reward, new_state, next_action)
            
            # 5. Store S' and A' for the *next* loop iteration
            sarsa_s_a[agent_turn]['state'] = new_state
            sarsa_s_a[agent_turn]['action'] = next_action

            # --- Stats and Terminal State Check ---
            self.first_policy_switch_run = first_policy_switch_run # Store this for plotting
            current_run_steps = self._track_stats_and_reset(step, current_run_steps)
            
            # If reset, we must re-initialize S-A for SARSA
            if current_run_steps == 0:
                for i in [0, 1]:
                    c = self.controllers[i]
                    s = c.get_current_state()
                    p = self.world.get_possible_actions(c.agent, c.other_agent)
                    a = c.choose_action(current_policy, p)
                    sarsa_s_a[i] = {'state': s, 'action': a}
            
            current_run_steps += 1

    def _track_stats_and_reset(self, step, current_run_steps):
        """Internal helper to track stats and reset world if terminal."""
        # Record Manhattan distance (once per step, e.g., after M moves)
        if step % 2 == 1:
            dist = abs(self.agent_f.x - self.agent_m.x) + abs(self.agent_f.y - self.agent_m.y)
            self.all_manhattan_distances.append(dist)
            
        if self.world.is_terminal_state():
            print(f"--- Step {step}: Terminal state {int(self.terminal_states_reached) + 1} reached in {current_run_steps + 1} steps! ---")
            self.terminal_states_reached += 1
            self.steps_per_run.append(current_run_steps + 1)
            
            # Reset world and agents
            self.world.reset()
            self.agent_f.reset(**AGENT_F_START)
            self.agent_m.reset(**AGENT_M_START)
            return 0 # Return 0 to reset current_run_steps
            
        return current_run_steps # Not terminal, return current count
        
    def _print_results(self):
        """Prints final stats and calls visualization functions."""
        print("\n--- Simulation Finished ---")
        print(f"Experiment: {self.config['name']}")
        print(f"Total Steps: {self.config['total_steps']}")
        print(f"Total Reward (F): {self.total_rewards[self.agent_f.name]}")
        print(f"Total Reward (M): {self.total_rewards[self.agent_m.name]}")
        print(f"Total Terminal States: {int(self.terminal_states_reached)}")
        if self.steps_per_run:
            print(f"Avg steps per run: {sum(self.steps_per_run) / len(self.steps_per_run):.2f}")
        if self.all_manhattan_distances:
            avg_dist = sum(self.all_manhattan_distances) / len(self.all_manhattan_distances)
            print(f"Average Manhattan Distance: {avg_dist:.2f}")
        print(f"Q-Table size (F): {len(self.controller_f.q_table)} states")
        print(f"Q-Table size (M): {len(self.controller_m.q_table)} states")

        # --- Call Visualizations ---
        Visualization.plot_performance(
            self.steps_per_run, 
            getattr(self, 'first_policy_switch_run', -1), 
            self.config['name']
        )
        
        if self.config.get("visualize_paths", False):
            print("\nGenerating final path visualizations...")
            other_agent_start_pos = (AGENT_M_START['x'], AGENT_M_START['y'])
            
            Visualization.plot_attractive_paths(
                self.controller_f.q_table, 
                agent_has_block=False, 
                other_agent_pos=other_agent_start_pos,
                title=f"{self.config['name']}\nAgent F Paths (NO Block)"
            )
            Visualization.plot_attractive_paths(
                self.controller_f.q_table, 
                agent_has_block=True, 
                other_agent_pos=other_agent_start_pos,
                title=f"{self.config['name']}\nAgent F Paths (WITH Block)"
            )
        else:
            print("\nSkipping path visualization for this run.")
        
        Visualization.print_q_table_sample(self.controller_f.q_table, num_states=5)