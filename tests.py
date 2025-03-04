from envs.simple_dungeonworld_env import DungeonMazeEnv
import numpy as np

SIZE=8

# Load the simple dungeon maze env
env = DungeonMazeEnv(render_mode="human", grid_size=SIZE)

# Check the same seed returns the same maze
env.reset(seed=124)
maze1 = env.maze
env.reset(seed=124)
maze2 = env.maze
assert maze1.__eq__(maze2)

# Check a different seed returns a different maze
env.reset(seed=2)
maze3 = env.maze
assert not maze1.__eq__(maze3)

obs = env.reset(seed=124)
env.render()

# Check target in the correct place
assert env.maze.get_grid_item(SIZE-2, SIZE-2).type == 'target'
assert np.array_equal(146*np.ones((20,20)), env.maze.get_grid_item(SIZE-2, SIZE-2).image)
