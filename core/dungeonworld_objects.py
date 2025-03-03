from PIL import Image
import numpy as np


class MazeObject:
    """
    Base class for any object or entity found within the maze.
    """

    def __init__(self, type, pos):
        self.type = type
        self.pos = pos
        self.image = None

    def can_overlap(self):
        """
        Boolean indicating whether robot and object can occupy the same cell.
        """
        return False

    def get_camera_view(self):
        """
        Returns the 'camera view' of the robot when facing cell.
        """
        return self.image

    def can_be_killed_by_sword(self):
        """
        Boolean indicating whether object can be killed (i.e. removed from the environment)
        by robot using a sword.
        """
        return False

    def can_be_killed_by_bow(self):
        """
        Boolean indicating whether object can be killed (i.e. removed from the environment)
        by robot using a bow.
        """
        return False


class Target(MazeObject):
    """
    Robot's target or goal location.
    """

    def __init__(self, pos):
        super().__init__("target", pos)
        # Target appears as greyscale green image - update to door image?
        self.image = 146 * np.ones((20, 20))

    def can_overlap(self):
        return True


class Wall(MazeObject):
    """
    The maze walls, robot cannot walk through them.
    """

    def __init__(self, pos):
        super().__init__("wall", pos)
        # Walls appear as black image.
        self.image = np.zeros((20, 20))


class Orc(MazeObject):
    """
    Orc creatures.
    Can be killed with a sword but not a bow as too strong.
    Will kill robot if overlapped.
    """

    def __init__(self, pos):
        super().__init__("orc", pos)
        im = Image.open("images/orc.png")
        self.image = np.array(im)
        im.close()

    def can_overlap(self):
        return True

    def can_be_killed_by_sword(self):
        return True


class Wingedbat(MazeObject):
    """
    Winged bat creatures.
    Can be killed with a bow but not a sword as flying.
    Will kill robot if overlapped.
    """

    def __init__(self, pos):
        super().__init__("wingedbat", pos)
        im = Image.open("images/wingedbat.png")
        self.image = np.array(im)
        im.close()

    def can_overlap(self):
        return True

    def can_be_killed_by_bow(self):
        return True


class Lizard(MazeObject):
    """
    Lizard creatures.
    Can be killed with both bow and sword.
    Will kill robot if overlapped.
    """

    def __init__(self, pos):
        super().__init__("lizard", pos)
        im = Image.open("images/lizard.png")
        self.image = np.array(im)
        im.close()

    def can_overlap(self):
        return True

    def can_be_killed_by_bow(self):
        return True

    def can_be_killed_by_sword(self):
        return True
