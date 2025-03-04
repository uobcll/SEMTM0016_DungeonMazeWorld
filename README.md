# SEMTM0016_DungeonMazeWorld

## Overview

Simple implementation of the maze solving gridworld problem based on [Gymnasium](https://gymnasium.farama.org/index.html) and heavily inspired by [Minigrid](https://minigrid.farama.org/index.html).

## How do you use it?

## Default MDP (`DungeonMazeEnv` class)

Action Space: The action space is discrete in the range `{0,2}` for `{turn left, turn right, move forwards}`.

Observation Space: The observation space is a composite with:

- The robot's (x,y) postion encoded as an element of `{0, ..., size-1}^2`.
- The robot's cardinal direction encoded as an integer `{0, ..., 3}`.
- The robot's 'camera view' encoded as a 20x20 pixel greyscale image, `{0, ..., 255}^(20x20)`.
- The target (x,y) postion encoded as an element of `{0, ..., size-1}^2`.

Starting State: The episode starts with the robot facing south in position `[1,1]` and the target in position `[grid_size-2, grid_size-2]` which align with the entrance and exit of the randomly generated maze. To use the same maze for each episode call `env.reset(seed=some_seed_value)`. New classes can be made to allocate different starting positions for the robot and target.

Transitions: Always deterministic.

Rewards: Every action incurs a -1 reward, except the action `move forwards` when the cell in front contains a non-overlappable object incurs a -100 reward.

Episode end: By default, an episode ends if the agent's position matches the target position or the robot crashes (i.e. tries to move to a cell with a non-overlappable object).
