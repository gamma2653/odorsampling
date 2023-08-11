# Glomeruli and Mitral Cell Objects
# Mitchell Gronowitz
# Spring 2015

# Edited by Christopher De Jesus
# Summer 2023

from __future__ import annotations

from collections import Counter
import logging
import builtins
from abc import ABC
# Used for asserts
if builtins.__debug__:
    from numbers import Real

from odorsampling import config, utils

# Type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import MutableMapping, Optional
    from RnO import Receptor
    

logger = logging.getLogger(__name__)
config.default_log_setup(logger)

cell_counter = Counter()
"""
Maps cell type to the number of cells of that type that have been created.
"""

# TODO: Add multithreading support
# FIXME: Fix `start_0`.
def add_count(cell_type: type[Cell], start_0: bool = True) -> None:
    """
    Increments the number of cells of the given type that have been created.
    
    Parameters
    ----------
    cell_type
        The Cell type who's ID to increment.
    start_0
        WIP, low-priority; An optional boolean argument whether to start IDs from 0 or 1.
        True by default. Currently, if set to False, may break the program.
    """
    cell_counter[cell_type] += 1
    return cell_counter[cell_type] - start_0

def reset_count(cell_type: type[Cell]):
    """
    Resets the counter for the given type.
    """
    del cell_counter[cell_type]

# TODO: consider making these dataclasses

class Cell(ABC):
    """
    Abstract class to be extended by `Glom` and `Mitral`.
    """
    def _add_count(self, start_0: bool = True):
        return add_count(self.__class__, start_0=start_0)

    @property
    def id(self) -> int:
        """
        ID of the cell.
        """
        return self._id
    
    @id.setter
    def id(self, value: Optional[int]) -> None:
        """
        If set to None, will assign a new ID using add_count. 
        """
        if value is None:
            self._id = self._add_count()
        else:
            self._id = int(value)

    @property
    def activ(self) -> float:
        """
        Returns activation level of the Glom.
        """
        return self._activ

    @activ.setter
    def activ(self, value: float) -> None:
        """
        Rounds value and sets it to activation level.

        Parameters
        ----------
        value
            Value is a float between 0 and 1.
        """
        N_DIGITS = 6
        assert value <= 1 and value >= 0, "Not between 0 and 1"
        self._activ = round(float(value), N_DIGITS)

    @property
    def loc(self) -> tuple[Real, Real]:
        """
        Returns location of the glom.
        """
        return self._loc

    @loc.setter
    def loc(self, value: tuple[Real, Real]) -> None:
        """
        Sets value to loc.

        Parameters
        ----------
        value
            is a 2D list of numbers
        """
        # FIXME: ensure tuple
        assert len(value) == 2 and isinstance(value[0], Real), "Not a 2D list of numbers!"
        self._loc = tuple(value)
    
    def __init__(self, id_: Optional[int], activ: float, loc: tuple[Real, Real]) -> None:
        """
        Initializes a cell with an id, activation level, and location.

        Parameters
        ----------
        id_ - Optional[int]
            the ID to assign the cell.
        activ - float
            The initial activation level of the cell.
        loc - tuple[Real, Real]
            The initial location of the cell.
        """
        self.id = id_
        self.activ = activ
        self.loc = loc
        

# TODO: update numpy docstrings
class Glom(Cell):
    """
    Represents a glomerulus cell that communicates with a single receptor.

    Attributes
    ----------
    id : int
        Identifies the glomerulus cell.
    activ : float
        Between (0,1) - activation level of glomerulus cell.
    loc : Tuple[float, float]
        x,y coordinates of the glom on the surface of the Olfactory bulb
    dim : Tuple[float, float]
        row x columns 
    conn : int
        Number of mitral cells connected to
    _recConn : dict
        dict of connecting recs:weights
    """

    @property
    def dim(self) -> tuple[int]:
        """
        Returns the dimensions of the glom.
        """
        return self._dim

    @dim.setter
    def dim(self, value: tuple[int]) -> None:
        """
        Sets dim to value.
        
        Parameters
        ----------
        value
            A length 2 list of numbers.
        """
        assert len(value) == 2 and isinstance(value[0], int), "Not a 2D list of numbers!"
        self._dim = tuple(value)
        
    @property
    def conn(self) -> int:
        """
        Returns conn of glom.
        """
        return self._conn
    
    @conn.setter
    def conn(self, value: int) -> None:
        """
        Sets value to conn.
        
        Parameters
        ----------
        value - int
            Value to set conn.
        """
        self._conn = int(value)
    
    def setRecConn(self, value: dict) -> dict:
        """
        Sets value to recConn
        """
        self._recConn = dict(value)

    def addRecConn(self, key: dict, weight):
        """
        Sets value to recConn
        """
        logger.debug("Glom cell[%s] added receptivity connection: [%s]", self._id, key)
        self._recConn[key] = weight
    

    def __init__(self, id_: Optional[int], activ=0.0, loc=(0,0), dim=(0,0), conn=0):
        """
        Initializes the Glom object
        
        Parameters
        ----------
        id_
            The ID with which to initialize the glom cell.
        activ
            The initial activation level of the cell.
        loc
            The initial location of the cell.
        dim
            The initial dims of the cell.
        conn
            The initial conn of the cell.
        """
        super().__init__(id_, activ, loc)
        self._dim: tuple[float, float] = dim
        self._conn: int = conn
        # TODO: Took me some time to track down this type. Would be a circular import,
        # but possible using PEP 484#forward-references. Still, best to remove potentially
        # circular import when possible
        self._recConn: dict['Receptor', float] = {}

    def __str__(self):
        """Returns a Glomural object description with activation energy"""
        return f"Id: {self.id} activ: {self.activ}"

    # TODO: Double check if implemented correctly
    @staticmethod
    # def generate_random_loc(xLowerBound: int, xUpperBound: int, yLowerBound: int, yUpperBound: int) -> tuple[int, int]:
    def generate_random_loc(xBounds: tuple[int, int], yBounds: tuple[int, int]) -> tuple[int, int]:
        """
        Returns a random glom location.
        
        Parameters
        ----------
        xBounds
            Tuple of low-highs
        yBounds
            Tuple of low-highs
        """
        (xLowerBound, xUpperBound), (yLowerBound, yUpperBound) = xBounds, yBounds
        randomGlomX = utils.RNG.integers(xLowerBound, xUpperBound, endpoint=False)
        if randomGlomX == xLowerBound or randomGlomX == xUpperBound:
            randomGlomY = utils.RNG.integers(yLowerBound, yUpperBound, endpoint=False)
        else:
            randomGlomY = utils.RNG.choice([yLowerBound, yUpperBound])
        return randomGlomX, randomGlomY

class Mitral(Cell):
    """Represents a mitral cell that samples from glomeruli.
    Attributes
    ----------
    _id : int
        identifies the mitral cell
    _activ : float
        value between [0,1]: activation level of mitral cell
    _loc : list[float, float]
        coordinates of the mitral cell on the surface of the bulb
    _glom : dict
        where the keys are glom and the values are weights
    
    """
        
    @property
    def glom(self) -> MutableMapping[Glom, float]:
        """Returns dictionary of connected glom"""
        return self._glom

    @glom.setter
    def glom(self, value: MutableMapping[Glom, float]) -> None:
        """Sets glomeruli to value.
        Precondition: Value is a dict containing glomeruli id's and weights."""
        self._glom = dict(value)

    def __init__(self, id_: Optional[int], activ=0.0, loc: tuple[float, float]=(0.0,0.0), glom=None):
        """Initializes a Mitral cell"""
        super().__init__(id_, activ, loc)
        self.glom = {} if glom is None else glom

    def __str__(self):
        """Returns a Mitral object description with activation energy and ID's 
        of connected glomeruli."""
        # *********************** From Python 3.6 onwards, the standard dict type maintains insertion order by default *****************************
        gstring = self.glom.keys()
        return f"Mitral ID: {self.id} Mitral Activ: {self.activ} Glom: {gstring}"
