from envs.simple_dungeonworld_env import DungeonMazeEnv
import numpy as np

SIZE=8

# Load the simple dungeon maze env
env = DungeonMazeEnv(render_mode="human", grid_size=SIZE)

# Initialise with set seed
env.reset(seed=124)

# Manual control for checking
for i in range(100):
    action = input("input action, 0=right, 1=left, 2=forwards")
    next_state, reward, terminated, stuck, info = env.step(int(action))
    print("next_state", next_state)
    print("reward", reward)
    print("terminated", terminated)
