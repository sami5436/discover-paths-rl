# 2-Agent Reinforcement Learning (PD-World)

## Overview
This project is a Python implementation of "Using Reinforcement Learning To Discover Paths in a 2-Agent Transportation World."

It uses Q-Learning and SARSA algorithms to train two agents ('F' and 'M') to cooperatively solve a block pickup and dropoff problem in a 5x5 grid.  
The system runs a suite of experiments to test and analyze agent performance, coordination, and adaptability as specified in the project requirements.

---

## How to Run

### 1. Prerequisites
This project requires the following Python libraries:

- matplotlib
- numpy

You can install them using pip:

```bash
pip install numpy matplotlib
````

---

### 2. Running the Simulation

Please ensure all .py files are within the same directory when running this through the command line.

All experiments are defined and executed from main.py.

```bash
python main.py
```

You will not see any output in your console.
All simulation logs, including step-by-step results and final summaries, are saved to a timestamped `.txt` file in the `/results` folder.

When the simulation is complete, you will see a single message:

```
All simulation output saved to 'results/simulation_log_... .txt'
```

---

### 3. Viewing the Output

Navigate to the `/results` folder. You will find:

* `simulation_log_... .txt`: A complete text log of all experiment runs
* `...._performance_plot.png`: Line graphs showing the "Steps per Run" for each experiment
* `...._path_plot.png`: Visual grid plots showing the final "Attractive Paths" learned by the agents

---

## Program Structure and File Logic

The program is organized into 7 Python files, each with a specific responsibility:

---

### 1. `main.py` — The Entry Point

**High-Level Logic:**
This is the main script you execute.
Its only job is to define the list of all 14 experiments (the 4 core experiments, each run twice, plus the LR variations) and pass them one-by-one to the ExperimentRunner.
It’s the "conductor" that starts the show.

---

### 2. `constants.py` — Global Configuration

**High-Level Logic:**
This file is a "single source of truth" for all static experiment parameters.
It defines:

* Grid size
* Pickup/Dropoff locations
* Agent start positions
* Default algorithm parameters (like α, γ, and total steps)

This makes it easy to tweak the experiment setup in one place.

---

### 3. `logger.py` — Output Logging

**High-Level Logic:**
This file defines the Logger class.
It is instantiated once in `main.py` and is responsible for redirecting all `print()` statements from all other files into a single, timestamped `.txt` log file in the `/results` folder.

This keeps the console clean and creates a permanent, detailed record of every run.

---

### 4. `environment.py` — The World & Rules Engine

**High-Level Logic:**
This file defines the PDWorld class — the "game board" and "referee."
It knows nothing about Q-learning. Its job is to:

* Manage the grid state (block counts at P/D locations)
* Enforce all game rules (e.g., "can’t move off-grid," "can’t pick up if holding a block," "can’t move into the other agent’s space")
* Apply actions, calculate the correct reward (+13 or -1), and check for the terminal state (all blocks delivered)

---

### 5. `agent.py` — The Agents & Their Brains

**High-Level Logic:**
This file defines two classes:

* **Agent:**
  A simple data object that holds the "body" of an agent: its (x, y) position and `has_block` status.

* **RLAgentController:**
  The "brain" of the agent. Each agent gets its own controller.
  This class owns the agent’s Q-table (a dictionary) and is responsible for all learning logic:

  * Choosing an action based on the current policy (`PRANDOM`, `PGREEDY`, `PEXPLOIT`)
  * Implementing `update_q_table` (Q-Learning) and `update_sarsa_table` (SARSA) update formulas

---

### 6. `visualization.py` — The Graphing Engine

**High-Level Logic:**
This class contains all matplotlib code.
It has no knowledge of the simulation itself — it simply takes processed data from the ExperimentRunner and generates the two required visual outputs:

* `plot_performance()`: Creates the “Steps per Run” line graph
* `plot_attractive_paths()`: Creates the 5x5 grid with arrows (a quiver plot) to visualize the agents' learned paths

---

### 7. `experiment.py` — The Lab Manager

**High-Level Logic:**
This is the most complex class. The ExperimentRunner does the “science.” It:

* Initializes the world and agent “brains” based on a configuration dictionary from `main.py`
* Runs the main 8000-step simulation loop
* Handles the policy schedule (e.g., switching from `PRANDOM` to `PEXPLOIT` at step 500)
* Handles the "world change" for Experiment 4
* Calls the correct `update_q_table` or `update_sarsa_table` function
* Records all metrics (steps per run, rewards, Manhattan distance)
* At the end, calls the Visualization class to save the final graphs

---

## Output Summary

At the end of each simulation, you will get:

* Detailed experiment logs (`.txt`)
* Performance plots (`.png`)
* Learned path visualizations (`.png`)

All files are automatically saved inside the `/results` folder.

---

## Algorithms Used

* Q-Learning
* SARSA 

Both agents learn cooperatively to minimize the number of steps required to transport all blocks while adhering to grid and interaction constraints.