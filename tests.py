from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions, Directions
import numpy as np

SIZE=8
EMPTY_CELL_IMAGE = np.ones((20,20)) * 255
WALL_CELL_IMAGE = np.zeros((20,20))
TARGET_CELL_IMAGE = np.ones((20,20)) * 146

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

env.reset(seed=124)

# Check target in the correct place
assert env.maze.get_cell_item(SIZE-2, SIZE-2).type == 'target'
assert np.array_equal(TARGET_CELL_IMAGE, env.maze.get_cell_item(SIZE-2, SIZE-2).image)

# Check robot initialises correctly
assert np.array_equal(env.robot_position, np.array([1,1]))
assert env.robot_direction == Directions.south
assert np.array_equal(env.robot_camera_view, EMPTY_CELL_IMAGE)

# Check turn left works
observation, reward, terminated, truncated, info = env.step(Actions.turn_left)
assert np.array_equal(observation["robot_position"], np.array([1,1]))
assert observation["robot_direction"] == Directions.east
assert np.array_equal(observation["robot_camera_view"], WALL_CELL_IMAGE)
assert reward == -1
assert terminated == False

# Check turn right works
observation, reward, terminated, truncated, info = env.step(Actions.turn_right)
assert np.array_equal(observation["robot_position"], np.array([1,1]))
assert observation["robot_direction"] == Directions.south
assert np.array_equal(observation["robot_camera_view"], EMPTY_CELL_IMAGE)
assert reward == -1
assert terminated == False

# Check move forwards works
observation, reward, terminated, truncated, info = env.step(Actions.move_forwards)
env.render()
assert np.array_equal(observation["robot_position"], np.array([1,2]))
assert observation["robot_direction"] == Directions.south
assert np.array_equal(observation["robot_camera_view"], WALL_CELL_IMAGE)
assert reward == -1
assert terminated == False

# Check move forwards again, crash into wall
observation, reward, terminated, truncated, info = env.step(Actions.move_forwards)
assert np.array_equal(observation["robot_position"], np.array([1,2]))
assert observation["robot_direction"] == Directions.south
assert np.array_equal(observation["robot_camera_view"], WALL_CELL_IMAGE)
assert reward == -100
assert terminated == True

# Check action sequence to reach target
env.reset(seed=124)
total_reward = 0
action_sequence = [Actions.move_forwards,
                   Actions.turn_left,
                   Actions.move_forwards,
                   Actions.move_forwards,
                   Actions.turn_right, 
                   Actions.move_forwards,
                   Actions.move_forwards,
                   Actions.turn_left,
                   Actions.move_forwards,
                   Actions.move_forwards,
                   Actions.turn_right,
                   Actions.move_forwards, 
                   Actions.move_forwards]

for action in action_sequence:
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    assert terminated == False
    assert truncated == False

# Check we can see the target correctly
observation, reward, terminated, truncated, info = env.step(Actions.turn_left)
total_reward += reward
assert np.array_equal(observation["robot_camera_view"], TARGET_CELL_IMAGE)

# Check we terminate at the target state
observation, reward, terminated, truncated, info = env.step(Actions.move_forwards)
total_reward += reward
assert np.array_equal(observation["robot_position"], observation["target_position"])
assert terminated == True
assert total_reward == -15

