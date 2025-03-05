import numpy as np

from .dungeonworld_objects import Target, Wall, Orc, Wingedbat, Lizard

def generate_maze(size, np_rng=None, seed=None):
    """
    Maze generation using iterative randomised DFS from https://en.wikipedia.org/wiki/Maze_generation_algorithm
    Note that mazes have at least a one cell buffer wall around all walkable cells.
    """
    # Minimum size of maze is 6x6
    assert size >= 6

    # Make sure dimensions are even to account for both 
    # boundary buffer walls and using cells as walls
    assert size % 2 == 0

    if np_rng is None:
        np_rng = np.random.default_rng(seed=seed)
    
    # Create initial grid filled with walls, 
    # reserve buffer for entrance/exit
    maze = np.ones((size-1, size-1))

    # Calculate number of aisles that are not walls
    num_aisles = (size-1) // 2

    # Initialise with starting point, account for outer wall
    start_x, start_y = (0, 0)
    maze[2*start_x+1, 2*start_y+1]=0
    stack = [(start_x, start_y)]

    # Define possible directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while len(stack) > 0:
        current_x, current_y = stack[-1]
        
        # Shuffle directions and search for unvisited neighbour
        np_rng.shuffle(directions)
        for dx, dy in directions:
            neighbour_x, neighbour_y = current_x + dx, current_y + dy
            if (neighbour_x >= 0 and neighbour_y >= 0 and neighbour_x < num_aisles 
                and neighbour_y < num_aisles and maze[2*neighbour_x+1, 2*neighbour_y+1]==1):
                # Next cell in maze
                maze[2*neighbour_x+1, 2*neighbour_y+1] = 0
                # Connecting path through wall
                maze[2*current_x+1+dx, 2*current_y+1+dy] = 0
                # Add new cell to stack
                stack.append((neighbour_x, neighbour_y))
                break
        else: # no break
            stack.pop()

    # Add the extra buffer walls for protruding entrance and exit.
    maze = np.pad(maze, ((0, 1), (1, 0)), "constant", constant_values=1)

    # Set the entrance and exit
    maze[1, 1] = 0
    maze[-2, -2] = 0

    return maze
    
class MazeGrid:
    """
    MazeGrid object for representing grid and operations on it.
    """
    # Map of object type to integers
    OBJECT_TO_IDX = {
        'empty': 0,
        'wall' : 1,
        'target': 2,
        'orc': 3,
        'wingedbat': 4,
        'lizard': 5,
    }

    # A flipped version of OBJECT_TO_IDX where the keys and values have been switched
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

    def __init__(self, size, empty=True, np_rng=None):
        """Set up the maze.
        
        If `empty` is True then just create an empty grid.
        Otherwise generate the maze, add walls and add a target.

        If `np_rng` is provided, this will be used as the seed for the rng.
        """
        # Maze is always square
        self.width = size
        self.height = size

        # Initialise empty grid
        self.grid = [None] * size * size

        # if we have requested an empty maze, then there's nothing more to do and we can exit the
        # method early
        if empty:
            return
        
        # Otherwise, generate the maze, add the walls and add the target
        
        # Generate the maze
        maze = generate_maze(size, np_rng)

        # Add the walls to grid
        for (x, y), elem in np.ndenumerate(maze):
            if elem == 1:
                self.add_cell_item(x, y, Wall(pos=np.array([x, y])))

        # Add the target at the maze exit (always at [-2, -2])
        self.add_cell_item(self.width-2, self.height-2, Target(pos=np.array([-2, -2])))

    def __eq__(self, other):
        """Allows us to compare mazes to one another."""
        grid1 = self.encode_maze_to_array()
        grid2 = other.encode_maze_to_array()
        return np.array_equal(grid2, grid1)

    def add_cell_item(self, x, y, maze_object):
        """Add maze object to grid at the specified x y coordinates."""
        assert x>=0 and x<self.width
        assert y>=0 and y<self.height
        self.grid[y * self.width + x] = maze_object

    def get_cell_item(self, x, y):
        """Retrieve an item from the grid at the specified x y coordinates."""
        assert x>=0 and x<self.width
        assert y>=0 and y<self.height
        return self.grid[y * self.width + x]

    def encode_maze_to_array(self):
        """
        Produces the entire grid as a encoded numpy array.
        """
        array = np.zeros((self.width, self.height), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                maze_object = self.get_cell_item(i, j)
                if maze_object is None:
                    array[i,j] = self.OBJECT_TO_IDX['empty']
                else:
                    array[i,j] = self.OBJECT_TO_IDX[maze_object.type]
        return array
    
    @staticmethod
    def decode_maze_from_array(array):
        """
        Produces the grid for the maze from an encoded array.

        E.g. An encoded array for grid size 6 could look like,

        [[1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 2, 1],
        [1, 1, 1, 1, 1, 1]]
        
        where 0 indicates an empty cell, 1 indicates a wall object 
        in the cell, and 2 indicates the target object in the cell.
        """
        width, height = array.shape
        assert width == height
        
        maze = MazeGrid(width)
        for i in range(width):
            for j in range(height):
                maze_object_type_index =array[i,j]
                if maze_object_type_index == MazeGrid.OBJECT_TO_IDX['empty']:
                    continue

                maze_object_type = MazeGrid.IDX_TO_OBJECT[maze_object_type_index]

                if maze_object_type == 'wall':
                    maze_object = Wall(pos=np.array([i, j]))
                elif maze_object_type == 'target':
                    maze_object = Target(pos=np.array([i, j]))
                elif maze_object_type == 'orc':
                    maze_object = Orc(pos=np.array([i, j]))
                elif maze_object_type == 'wingedbat':
                    maze_object = Wingedbat(pos=np.array([i, j]))
                elif maze_object_type == 'lizard':
                    maze_object = Lizard(pos=np.array([i, j]))
                else:
                    assert False, f"Unknown maze object type in decode {maze_object_type}"

                maze.add_cell_item(i, j, maze_object)
        return maze


