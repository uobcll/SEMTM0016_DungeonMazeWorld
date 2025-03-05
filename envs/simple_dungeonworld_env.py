"""
Simple HeroBot and the MazeDungeon Environment.
"""
from enum import IntEnum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from core.dungeonworld_grid import MazeGrid

class Actions(IntEnum):
    # Enumeration of possible actions
    # Turn right, turn left, move forwards
    right = 0
    left = 1
    forwards = 2

class Directions(IntEnum):
    # Enumeration of cardinal directions the robot can face
    # taking north as top of maze
    north = 0
    east = 1
    south = 2
    west = 3

class DungeonMazeEnv(gym.Env):
    """
    2D maze grid world environment for robot.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size=16):
        """TODO: Add docstring outlining what this does."""
        self.grid_size = grid_size
        self.window_size = 512

        # We have 3 actions, corresponding to "turn right", "turn left", "move forwards"
        self.action_space = spaces.Discrete(len(Actions))
        
        # Observations are dictionaries with:
        # The robot postion encoded as an element of {0, ..., size-1}^2,
        # The robot direction encoded as an integer {0, ..., 4},
        # The robot camera view encoded as a dummy 20x20 pixel greyscale image, {0, ..., 255}^(20x20)
        # The target postion encoded as an element of {0, ..., size-1}^2.
        self.observation_space = spaces.Dict(
            {
                "robot_position": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
                "robot_direction": spaces.Discrete(len(Directions)),
                "robot_camera_view": spaces.Box(low=0, high=255, shape=(20, 20), dtype=np.int32),
                "target_position": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        
        self.window = None
        self.clock = None

    def get_observations(self):
        """TODO: Add docstring saying what an observation is."""
        return {
            "robot_position": self.robot_position,
            "robot_direction": self.robot_direction,
            "robot_camera_view": self.robot_camera_view,
            "target_position": self.target_position, 
        }
    
    def get_robot_direction_vector(self):
        """
        Get the direction vector for the robot, pointing in the direction
        of forward movement.
        """
        direction_vectors = [
            # Up (negative Y)
            np.array((0, -1)),
            # Pointing right (positive X)
            np.array((1, 0)),
            # Down (positive Y)
            np.array((0, 1)),
            # Pointing left (negative X)
            np.array((-1, 0)),
        ]
        assert self.robot_direction >= 0 and self.robot_direction < 4
        return direction_vectors[self.robot_direction]

    def get_robot_front_pos(self):
        """
        Get the position of the cell that is right in front of the robot
        """
        return self.robot_position + self.get_robot_direction_vector()
    
    def get_robot_camera_view(self):
        """TODO: Add docstring explaining what the robot camera view is."""
        # Get the position in front of the robot
        position_in_front = self.get_robot_front_pos()

        # Get the contents of the cell in front of the agent
        cell_in_front = self.maze.get_grid_item(*position_in_front)

        if cell_in_front is None:
            # if nothing in front return a white image
            return np.ones((20,20))*255
        else:
            return cell_in_front.get_camera_view()

    def reset(self, seed=None, options=None):
        """TODO: Add docstring explaining what this does."""
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Create the grid capturing the maze as walls
        self.maze = MazeGrid(size=self.grid_size, empty=False, np_rng=self.np_random)

        # Set the target location
        self.target_position = np.array([self.grid_size-2, self.grid_size-2])

        # Set the robot's location, direction, inital camera view
        self.robot_position = np.array([1, 1])
        self.robot_direction = Directions.south
        self.robot_camera_view = self.get_robot_camera_view()

        # Update the observations
        observation = self.get_observations()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}     
    
    def step(self, action):
        """TODO: Add docstring explaining what this does."""
        reward = -1
        terminated = False

        # Get the position in front of the robot
        position_in_front = self.get_robot_front_pos()

        # Get the contents of the cell in front of the agent
        cell_in_front = self.maze.get_grid_item(*position_in_front)

        # Attempt actions
        if action == Actions.left:
            self.robot_direction -= 1
            if self.robot_direction < 0:
                self.robot_direction += 4
        elif action == Actions.right:
            self.robot_direction += 1
            if self.robot_direction > 3:
                self.robot_direction -= 4
        elif action == Actions.forwards:
            if cell_in_front is None or cell_in_front.can_overlap():
                self.robot_position = position_in_front
            else:
                # Terminate with penalty as robot tried to crash into an object in the cell in front.
                terminated = True
                reward = -100
        else:
            assert False, "unknown action"

        # Update the robot's camera view
        self.robot_camera_view = self.get_robot_camera_view()

        # Update the observations
        observation = self.get_observations()

        # An episode is terminated if the agent has reached the target
        if np.array_equal(self.robot_position, self.target_position):
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.target_position,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the walls
        for cell in self.maze.grid:
            if cell is not None and cell.type == 'wall':
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    pygame.Rect(
                        pix_square_size * cell.pos,
                        (pix_square_size, pix_square_size),
                    ),
                )

        # Now we draw the robot with direction it's facing
        if self.robot_direction == Directions.north:
            pygame.draw.polygon(
                canvas,
                (0, 0, 255),
                ((self.robot_position + np.array([0.1, 0.9])) * pix_square_size, 
                (self.robot_position + np.array([0.9, 0.9])) * pix_square_size,
                (self.robot_position + np.array([0.5, 0.1])) * pix_square_size),
            )
        elif self.robot_direction == Directions.east:
            pygame.draw.polygon(
                canvas,
                (0, 0, 255),
                ((self.robot_position + np.array([0.1, 0.9])) * pix_square_size, 
                (self.robot_position + np.array([0.1, 0.1])) * pix_square_size,
                (self.robot_position + np.array([0.9, 0.5])) * pix_square_size),
            )
        elif self.robot_direction == Directions.south:
            pygame.draw.polygon(
                canvas,
                (0, 0, 255),
                ((self.robot_position + np.array([0.9, 0.1])) * pix_square_size, 
                (self.robot_position + np.array([0.1, 0.1])) * pix_square_size,
                (self.robot_position + np.array([0.5, 0.9])) * pix_square_size),
            )
        elif self.robot_direction == Directions.west:
            pygame.draw.polygon(
                canvas,
                (0, 0, 255),
                ((self.robot_position + np.array([0.9, 0.1])) * pix_square_size, 
                (self.robot_position + np.array([0.9, 0.9])) * pix_square_size,
                (self.robot_position + np.array([0.1, 0.5])) * pix_square_size),
            )

        # Finally, draw some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
