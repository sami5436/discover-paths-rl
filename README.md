# discover-paths-

2d grid: pickup locations and dropoff locations

goal: 2 agents need  to move blocks from the 'P' locations to the 'D' locations until the task is complete (maybe all 'P' locations are empty or 'D' locations are full). == reaching a "terminal state".

agents: 2 agents, 'M' (male) and 'F' (female), who work at the same time

----

main things we needa pay attention to: 

- Blockage:  2 agents cannot be in the same position at the same time

- Coordination: blockage creates a major problem. we needa analyze if the agents learn to "divide the transportation task intelligently" or if they just get in each other's way

----

implementation: 

we can either do both or one of the options below for storing our q values (these are the values computed by q learning and sarsa - theres a formula). if we do both, we get extra credit

- Separate Q-Tables: Each agent learns individually but can see the other agent's position

- Single Q-Table: One "master" algorithm controls both agents simultaneously

----

experimentation: 

we need to include the following in our source code too...

- Experiment 1: Compare three different action-selection policies (PRANDOM, PGREEDY, PEXPLOIT) to see which one learns best

- Experiment 2: Compare the performance of the Q-learning algorithm against the SARSA algorithm.

- Experiment 3: Analyze how changing the learning rate to 0.15 and 0.45 affects system performance. 

- Experiment 4: Test adaptability. After the agents have learned the paths, you will change the pickup locations mid-experiment and analyze how well they "unlearn" the old, obsolete paths and find the new ones.