# Glomeruli and Mitral Cell Objects
# Mitchell Gronowitz
# Spring 2015

# Edited by Christopher De Jesus
# Summer 2023

from __future__ import annotations

import logging

import config

# Used for asserts
from numbers import Real
from typing import Sequence

# Type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import MutableMapping, Optional
    from RnO import Receptor

logger = logging.getLogger(__name__)
config.default_log_setup(logger)

# TODO: consider making these dataclasses

# TODO: update numpy docstrings
class Glom:
    """
    Represents a glomerulus cell that communicates with a single receptor.

    Attributes
    ----------
    _id : int
        Identifies the glomerulus
    _activ : float
        Between (0,1) - activation level of glomerulus
    _loc : Tuple[float, float]
        x,y coordinates of the glom on the surface of the Olfactory bulb
    _dim : Tuple[float, float]
        row x columns 
    _conn : int
        Number of mitral cells connected to
    _recConn : dict
        dict of connecting recs:weights
    """

    @property
    def id(self) -> int:
        return self._id
    
    @id.setter
    def id(self, value: int) -> None:
        """Sets value to id.
        Precondition: value is an integer"""
        value = int(value)
        old = getattr(self, '_id', None)
        logger.debug("Glom cell id changed: [%s->%s]", old, value)
        self._id = value    

    @property
    def activ(self) -> float:
        """Returns activation level of Glom."""
        return self._activ

    @activ.setter
    def activ(self, value: float) -> None:
        """Rounds value and sets it to activation level.
        Precondition: Value is a float between 0 and 1."""
        N_DIGITS = 6
        value = round(float(value), N_DIGITS)
        assert value <= 1 and value >= 0, "Not between 0 and 1"
        old = getattr(self, '_activ', None)
        logger.debug("Glom cell[%s] activ level changed: [%s->%s]", self._id, old, value)
        self._activ = value

    @property
    def loc(self) -> tuple[Real, Real]:
        """Returns location of glom"""
        return self._loc

    @loc.setter
    def loc(self, value: tuple[Real, Real]) -> None:
        """Sets value to loc.
        Precondition: value is a 2D list of numbers"""
        # FIXME: ensure tuple
        value = tuple(value)
        old = getattr(self, '_loc', None)
        assert len(value) == 2 and isinstance(value[0], Real), "Not a 2D list of numbers!"
        logger.debug("Glom cell[%s] loc changed: [%s->%s]", self._id, old, value)
        self._loc = value

    @property
    def dim(self) -> tuple[int]:
        """Returns dimensions of glom"""
        return self._dim

    @dim.setter
    def dim(self, value: tuple[int]) -> None:
        """Sets value to dim.
        Precondition: value is a 2D list of numbers"""
        value = tuple(value)
        assert len(value) == 2 and isinstance(value[0], int), "Not a 2D list of numbers!"
        old = getattr(self, '_dim', None)
        logger.debug("Glom cell[%s] dim changed: [%s->%s]", self._id, old, value)
        self._dim = value
        
    @property
    def conn(self) -> int:
        """Returns connections of glom"""
        return self._conn
    
    @conn.setter
    def conn(self, value: int) -> None:
        """Sets value to conn.
        Precondition: value is an int"""
        value = int(value)
        old = getattr(self, '_conn', None)
        logger.debug("Glom cell[%s] connectivity changed: [%s->%s]", self._id, old, value)
        self._conn = value
    
    def setRecConn(self, value: dict) -> dict:
        """Sets value to recConn"""
        value = dict(value)
        old = getattr(self, '_recConn', None)
        logger.debug("Glom cell[%s] receptor connectivity changed: [%s->%s]", self._id, old, value)
        self._recConn = value

    def addRecConn(self, key: dict, weight):
        """Sets value to recConn"""
        logger.debug("Glom cell[%s] added receptivity connection: [%s]", self._id, key)
        self._recConn[key] = weight
    

    def __init__(self, id_, activ=0.0, loc=(0,0), dim=(0,0), conn=0):
        """Initializes Glom object"""
        self._id: int = id_
        self._activ: float = activ
        self._loc: tuple[float, float] = loc
        self._dim: tuple[float, float] = dim
        self._conn: int = conn
        # TODO: Took me some time to track down this type. Would be a circular import,
        # but possible using PEP 484#forward-references. Still, best to remove potentially
        # circular import when possible
        self._recConn: dict['Receptor', float] = {}

    def __str__(self):
        """Returns a Glomural object description with activation energy"""
        return f"Id: {self.id} activ: {self.activ}"


class Mitral:
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
    def id(self) -> int:
        """Returns ID of mitral"""
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Sets value to id.
        Precondition: value is an integer"""
        value = int(value)
        old = getattr(self, '_id', None)
        logger.debug("Mitral cell[%s] id changed: [%s->%s]", old, old, value)
        self._id = value

    @property
    def activ(self) -> float:
        """Returns activation level of Mitral."""
        return self._activ

    @activ.setter
    def activ(self, value: float) -> None:
        """Rounds value and sets it to activation level.
        Precondition: Value is a float between 0 and 1."""
        N_DIGITS = 5
        value = round(float(value), N_DIGITS)
        assert value <= 1 and value >= 0, "Not between 0 and 1"
        old = getattr(self, '_activ', None)
        logger.debug("Mitral cell[%s] activ level changed: [%s->%s]", self._id, old, value)
        self._activ = value
        
    @property
    def loc(self) -> tuple[Real, Real]:
        """Returns location of mitral cell"""
        return self._loc

    @loc.setter
    def loc(self, value: tuple[Real, Real]) -> None:
        """Sets value to loc.
        Precondition: value is a 2D list of numbers"""
        value = tuple(value)
        assert len(value) == 2 and isinstance(value[0], Real), "Not a length: 2 tuple of numbers!"
        old = getattr(self, '_loc', None)
        logger.debug("Mitral cell[%s] loc changed: [%s->%s]", self._id, old, value)
        self._loc = value
        
    @property
    def glom(self) -> MutableMapping[Glom, float]:
        """Returns dictionary of connected glom"""
        return self._glom

    @glom.setter
    def glom(self, value: MutableMapping[Glom, float]) -> None:
        """Sets glomeruli to value.
        Precondition: Value is a dict containing glomeruli id's and weights."""
        value = dict(value)
        old = getattr(self, '_glom', None)
        logger.debug("Mitral cell[%s] glom map changed: [%s->%s]", self._id, old, value)
        self._glom = value

    def __init__(self, id_, activ=0.0, loc: tuple[float, float]=(0.0,0.0), glom=None):
        """Initializes a Mitral cell"""
        self.id = id_
        self.activ = activ
        self.loc = loc
        self.glom = {} if glom is None else glom

    def __str__(self):
        """Returns a Mitral object description with activation energy and ID's 
        of connected glomeruli."""
        # *********************** From Python 3.6 onwards, the standard dict type maintains insertion order by default *****************************
        gstring = self.glom.keys()
        return f"Mitral ID: {self.id} Mitral Activ: {self.activ} Glom: {gstring}"


