#Glomeruli and Mitral Cell Objects
#Mitchell Gronowitz
#Spring 2015

# Used for asserts
from numbers import Real

# Type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Tuple, MutableMapping

class Glom:
    """Represents a glomerulus cell that communicates with a single receptor.
    Instance attributes:
    _id    = [integer] identifies the glomerulus
    _activ = [float: in (0,1)] activation level of glomerulus
    _loc   = [Tuple: len = 2] x,y coordinates of the glom on the surface of the Olfactory bulb
    _dim   = [Tuple: len = 2] row x columns 
    _conn  = [integer] number of mitral cells connected to
    _recConn = [dict] dict of connecting recs:weights
    """

    # TODO: Revisit why these were checking exact type rather than isinstance

    @property
    def id(self) -> int:
        return self._id
    
    @id.setter
    def id(self, value: int) -> None:
        """Sets value to id.
        Precondition: value is an integer"""
        assert isinstance(value, int), "value is not an integer!!"
        self._id = value    

    @property
    def activ(self) -> float:
        """Returns activation level of Glom."""
        return self._activ

    @activ.setter
    def activ(self, value: float) -> None:
        """Rounds value and sets it to activation level.
        Precondition: Value is a float between 0 and 1."""
        assert isinstance(value, float), "Not a float!"
        assert value <= 1 and value >= 0, "Not between 0 and 1"
        self._activ = round(value, 6)

    @property
    def loc(self) -> "Tuple[Real]":
        """Returns location of glom"""
        return self._loc

    @loc.setter
    def loc(self, value: "Tuple[Real]") -> None:
        """Sets value to loc.
        Precondition: value is a 2D list of numbers"""
        assert isinstance(value, tuple), "value is not a list!!"
        assert len(value) == 2 and isinstance(value[0], Real), "Not a 2D list of numbers!"
        self._loc = value

    @property
    def dim(self) -> "Tuple[int]":
        """Returns dimensions of glom"""
        return self._dim

    @dim.setter
    def dim(self, value: "Tuple[int]") -> None:
        """Sets value to dim.
        Precondition: value is a 2D list of numbers"""
        assert isinstance(value, tuple), "value is not a list!!"
        assert len(value) == 2 and isinstance(value[0], int), "Not a 2D list of numbers!"
        self._dim = value
        
    @property
    def conn(self) -> int:
        """Returns connections of glom"""
        return self._conn
    
    @conn.setter
    def conn(self, value: int) -> None:
        """Sets value to conn.
        Precondition: value is an int"""
        assert isinstance(value, int)
        self._conn = value
    
    def setRecConn(self, value: dict) -> dict:
        """Sets value to recConn"""
        assert isinstance(value, dict), "value isn't a dictionary"
        self._recConn = value

    def addRecConn(self, value: dict, weight):
        """Sets value to recConn"""
        self._recConn[value] = weight
    

    def __init__(self, ID, activ=0.0, loc=(0,0), dim=(0,0), conn=0):
        """Initializes Glom object"""
        self.id = ID
        self.activ = activ
        self.loc = loc
        self.dim = dim
        self.conn = conn
        self._recConn = {}

    def __str__(self):
        """Returns a Glomural object description with activation energy"""
        return "Id: " + str(self.id) + " activ: " + str(self.activ)


class Mitral(object):
    """Represents a mitral cell that samples from glomeruli.
    Instance attributes:
    _id = [int] identifies the mitral cell
    _activ = [float: 0 - 1] activation level of mitral cell
    -loc = [2D list] coordinates of the mitral cell on the surface of the bulb
    _glom = dict where the keys are glom and the values are weights
    
    """
    @property
    def id(self) -> int:
        """Returns ID of mitral"""
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Sets value to id.
        Precondition: value is an integer"""
        assert isinstance(value, int), "value is not an integer!!"
        self._id = value

    @property
    def activ(self) -> float:
        """Returns activation level of Mitral."""
        return self._activ

    @activ.setter
    def activ(self, value: float) -> None:
        """Rounds value and sets it to activation level.
        Precondition: Value is a float between 0 and 1."""
        assert isinstance(value, float), "Not a float!"
        assert value <= 1 and value >= 0, "Not between 0 and 1"
        self._activ = round(value, 5)
        
    @property
    def loc(self) -> "Tuple[Real]":
        """Returns location of mitral cell"""
        return self._loc

    @loc.setter
    def loc(self, value: "Tuple[Real]") -> None:
        """Sets value to loc.
        Precondition: value is a 2D list of numbers"""
        assert isinstance(value, tuple), "value is not a list!!"
        assert len(value) == 2 and isinstance(value[0], Real), "Not a length: 2 list of numbers!"
        self._loc = value   
        
    @property
    def glom(self) -> "MutableMapping[Glom, dict]":
        """Returns dictionary of connected glom"""
        return self._glom

    @glom.setter
    def glom(self, value: "MutableMapping[Glom, dict]") -> None:
        """Sets glomeruli to value.
        Precondition: Value is a dict containing glomeruli id's and weights."""
        assert isinstance(value, dict), "Not a dict!"
        self._glom = value

    def __init__(self, ID, activ=0.0, loc=[0,0], glom={}):
        """Initializes a Mitral cell"""
        self.id = ID
        self.activ = activ
        self.loc = loc
        self.glom = glom

    def __str__(self):
        """Returns a Mitral object description with activation energy and ID's 
        of connected glomeruli."""
        # *********************** From Python 3.6 onwards, the standard dict type maintains insertion order by default *****************************
        gstring = self.glom.keys()
        return "Mitral ID: " + str (self.id) + " Mitral Activ: " + str(self.activ) + " Glom:" + str(gstring)


