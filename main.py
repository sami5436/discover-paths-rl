# this is discover-paths-rl/main.py
# this file contains the main function to run experiments
# It defines various experiment configurations and executes them.

import os
import sys
import datetime
import random
from logger import Logger
from experiment import ExperimentRunner

def main():
    """
    Defines all experiment configurations and runs them.
    You can comment/uncomment experiments to run them selectively.
    """
    
    # --- Base Parameters ---
    total_steps = 8000
    discount_factor = 0.5
    default_lr = 0.3
    
    # --- Experiment 1: Policy Comparison (Q-Learning) ---
    exp_1a = {
        "name": "Exp_1a_PRANDOM",
        "total_steps": total_steps,
        "algorithm": "Q_LEARNING",
        "learning_rate": default_lr,
        "discount_factor": discount_factor,
        "policy_schedule": [(total_steps, 'PRANDOM')],
        "visualize_paths": False 
    }
    
    exp_1b = {
        "name": "Exp_1b_PGREEDY",
        "total_steps": total_steps,
        "algorithm": "Q_LEARNING",
        "learning_rate": default_lr,
        "discount_factor": discount_factor,
        "policy_schedule": [(500, 'PRANDOM'), (total_steps - 500, 'PGREEDY')],
        "visualize_paths": False 
    }
    
    exp_1c = {
        "name": "Exp_1c_PEXPLOIT",
        "total_steps": total_steps,
        "algorithm": "Q_LEARNING",
        "learning_rate": default_lr,
        "discount_factor": discount_factor,
        "policy_schedule": [(500, 'PRANDOM'), (total_steps - 500, 'PEXPLOIT')],
        "visualize_paths": False 
    }
    
    # --- Experiment 2: Q-Learning vs SARSA ---
    exp_2 = {
        "name": "Exp_2_SARSA",
        "total_steps": total_steps,
        "algorithm": "SARSA", # Changed algorithm
        "learning_rate": default_lr,
        "discount_factor": discount_factor,
        "policy_schedule": [(500, 'PRANDOM'), (total_steps - 500, 'PEXPLOIT')],
        "visualize_paths": False # Don't plot for this run
    }
    
    # --- Experiment 3: Learning Rate Analysis ---
    exp_3_low_lr = {
        "name": "Exp_3_PEXPLOIT_LR_015",
        "total_steps": total_steps,
        "algorithm": "Q_LEARNING", 
        "learning_rate": 0.15, # Changed LR
        "discount_factor": discount_factor,
        "policy_schedule": [(500, 'PRANDOM'), (total_steps - 500, 'PEXPLOIT')],
        "visualize_paths": False
    }
    
    exp_3_high_lr = {
        "name": "Exp_3_PEXPLOIT_LR_045",
        "total_steps": total_steps,
        "algorithm": "Q_LEARNING", 
        "learning_rate": 0.45, 
        "discount_factor": discount_factor,
        "policy_schedule": [(500, 'PRANDOM'), (total_steps - 500, 'PEXPLOIT')],
        "visualize_paths": False 
    }
    
    # --- Experiment 4: Adaptability ---
    exp_4 = {
        "name": "Exp_4_Adaptability",
        "total_steps": total_steps,
        "algorithm": "Q_LEARNING",
        "learning_rate": default_lr,
        "discount_factor": discount_factor,
        "policy_schedule": [(500, 'PRANDOM'), (total_steps - 500, 'PEXPLOIT')],
        "visualize_paths": False 
    }


    
    experiments_to_run = [
        exp_1a,
        exp_1b,
        exp_1c,
        exp_2,
        exp_3_low_lr,
        exp_3_high_lr,
        exp_4,
        
        # --- Run 2 ---
        {**exp_1a, "name": "Exp_1a_PRANDOM_Run2", "visualize_paths": True},
        {**exp_1b, "name": "Exp_1b_PGREEDY_Run2", "visualize_paths": True},
        {**exp_1c, "name": "Exp_1c_PEXPLOIT_Run2", "visualize_paths": True},
        {**exp_2, "name": "Exp_2_SARSA_Run2", "visualize_paths": True},
        {**exp_3_low_lr, "name": "Exp_3_PEXPLOIT_LR_015_Run2", "visualize_paths": True},
        {**exp_3_high_lr, "name": "Exp_3_PEXPLOIT_LR_045_Run2", "visualize_paths": True},
        {**exp_4, "name": "Exp_4_Adaptability_Run2", "visualize_paths": True},
    ]
    
  

    for config in experiments_to_run:
        random.seed(random.randint(0, 100000)) 
        runner = ExperimentRunner(config)
        runner.run()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join('results', f"simulation_log_{timestamp}.txt")
    
    # Store the original stdout
    original_stdout = sys.stdout
    
    logger = Logger(log_filename)
    sys.stdout = logger
    sys.stderr = logger
    
    try:
        main()
    except Exception as e:
        print("\n" + "="*50)
        print(f"AN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
        print("="*50)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\nAll simulation output saved to '{log_filename}'")