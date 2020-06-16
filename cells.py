#Glomeruli and Mitral Cell Objects
#Mitchell Gronowitz
#Spring 2015

class Glom(object):
    """Represents a glomerulus cell that communicates with a single receptor.
    Instance attributes:
    _id    = [integer] identifies the glomerulus
    _activ = [float: 0 - 1] activation level of glomerulus
    _loc   = [2D list] x,y coordinates of the glom on the surface of the Olfactory bulb
    _dim   = [2D list] row x columns 
    _conn  = [integer] number of mitral cells connected to
    _recConn = [dict] dict of connecting recs:weights
    """

    def getId(self):
        """Returns ID of glom"""
        return self._id

    def setId(self, value):
        """Sets value to id.
        Precondition: value is an integer"""
        assert type(value) == int, "value is not an integer!!"
        self._id = value    

    def getActiv(self):
        """Returns activation level of Glom."""
        return self._activ

    def setActiv(self, value):
        """Rounds value and sets it to activation level.
        Precondition: Value is a float between 0 and 1."""
        assert type(value) == float, "Not a float!"
        assert value <= 1 and value >= 0, "Not between 0 and 1"
        self._activ = round(value, 6)

    def getLoc(self):
        """Returns location of glom"""
        return self._loc

    def setLoc(self, value):
        """Sets value to loc.
        Precondition: value is a 2D list of numbers"""
        assert type(value) == list, "value is not a list!!"
        assert len(value) == 2 and type(value[0]) in [int, float], "Not a 2D list of numbers!"
        self._loc = value

    def getDim(self):
        """Returns dimensions of glom"""
        return self._dim

    def setDim(self, value):
        """Sets value to dim.
        Precondition: value is a 2D list of numbers"""
        assert type(value) == list, "value is not a list!!"
        assert len(value) == 2 and type(value[0]) in [int], "Not a 2D list of numbers!"
        self._dim = value
        
    def getConn(self):
        """Returns connections of glom"""
        return self._conn
    
    def setConn(self, value):
        """Sets value to conn.
        Precondition: value is an int"""
        assert type(value) == int
        self._conn = value
    
    def setRecConn(self, value):
        """Sets value to recConn"""
        assert type(value) == dict, "value isn't a dictionary"
        self._recConn = value
        
    def addRecConn(self, value, weight):
        """Sets value to recConn"""
        self._recConn[value] = weight
    

    def __init__(self, ID, activ=0.0, loc=[0,0], dim=[0,0], conn=0):
        """Initializes Glom object"""
        self.setId(ID)
        self.setActiv(activ)
        self.setLoc(loc)
        self.setDim(dim)
        self.setConn(conn)
        self.setRecConn({})

    def __str__(self):
        """Returns a Glomural object description with activation energy"""
        return "Id: " + str(self.getId()) + " activ: " + str(self.getActiv())


class Mitral(object):
    """Represents a mitral cell that samples from glomeruli.
    Instance attributes:
    _id = [int] identifies the mitral cell
    _activ = [float: 0 - 1] activation level of mitral cell
    -loc = [2D list] coordinates of the mitral cell on the surface of the bulb
    _glom = dict where the keys are glom and the values are weights
    
    """
    def getId(self):
        """Returns ID of mitral"""
        return self._id

    def setId(self, value):
        """Sets value to id.
        Precondition: value is an integer"""
        assert type(value) == int, "value is not an integer!!"
        self._id = value

    def getActiv(self):
        """Returns activation level of Mitral."""
        return self._activ

    def setActiv(self, value):
        """Rounds value and sets it to activation level.
        Precondition: Value is a float between 0 and 1."""
        assert type(value) == float, "Not a float!"
        assert value <= 1 and value >= 0, "Not between 0 and 1"
        self._activ = round(value, 5)
        
    def getLoc(self):
        """Returns location of mitral cell"""
        return self._loc

    def setLoc(self, value):
        """Sets value to loc.
        Precondition: value is a 2D list of numbers"""
        assert type(value) == list, "value is not a list!!"
        assert len(value) == 2 and type(value[0]) in [int, float], "Not a 2D list of numbers!"
        self._loc = value   
        
    def getGlom(self):
        """Returns dictionary of connected glom"""
        return self._glom

    def setGlom(self, value):
        """Sets glomeruli to value.
        Precondition: Value is a dict containing glomeruli id's and weights."""
        assert type(value) == dict, "Not a dict!"
        self._glom = value

    def __init__(self, ID, activ=0.0, loc=[0,0], glom={}):
        """Initializes a Mitral cell"""
        self.setId(ID)
        self.setActiv(activ)
        self.setLoc(loc)
        self.setGlom(glom)

    def __str__(self):
        """Returns a Mitral object description with activation energy and ID's 
        of connected glomeruli."""
        # *********************** From Python 3.6 onwards, the standard dict type maintains insertion order by default *****************************
        g = []
        gstring = self.getGlom().keys()
        return "Mitral ID: " + str (self.getId()) + " Mitral Activ: " + str(self.getActiv()) + " Glom:" + str(gstring)


