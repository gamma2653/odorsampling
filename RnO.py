# Odor Spaces!
# Mitchell Gronowitz
# Spring 2014

# Edited by Christopher De Jesus
# Summer 2023

"""
QSpace, Odototope, Odorscene, and Receptor objects
Index of document:
1. Global variables
2. Defining all objects
3. Simple functions to create and activate each object
4. Experiments/Simulations utilizing objects
"""


from __future__ import annotations

import math
import random
import layers
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pylab
from matplotlib.backends.backend_pdf import PdfPages
import copy
import time
import functools

#from matplotlib import mlab, cm
from matplotlib.patches import Ellipse
import numpy.random as rnd

import params

# Used for asserts
from numbers import Real

# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
    # from typing import Tuple
    # from typing import List
    # ^Removed in favor of Python 3.9 list generics (see PEP 585)

###Global Variables
peak_affinity = -8     # literally 10e-8, not influenced by minimum_affinity value
minimum_affinity = 2   # asymptotic affinity exponent, negligible
m = 1 #Hill Coefficient
ODOR_REPETITIONS = 2 #Amount of odorscene repetitions to create a smooth graph
ANGLES_REP = 2

# SD_NUMBER = 1.5
# SD_NUMBER = params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION

# TODO: Move to params.py
###Global Variables when c!=1 (multiple conn between rec and glom)
glom_penetrance = .68  # primary glom:rec connection weight if c != 1
s_weights = [0.12,0.08,0.03,0.03,0.02,0.02,0.01,0.01] # The remaining glom:rec connection weights if c != 1
numRow = 6 # num of rows of glom
numCol = 5 # num of cols of glom  (numRow*numCol = total number of Glom)
constant_attachments = True


class QSpace:
    """Defines the size of the sample space
    ***ex: [(0,5),(0,5)] means qspace extends from 0 to 5 (not including 5) 

    Attributes
    ----------
    _size : list[tuple[float]]
        defines the size of the sample space"""
    
    @property
    def size(self) -> list[tuple[float, float]]:
        #Returns the odors.
        return self._size

    @size.setter
    def size(self, value: list[tuple[float, float]]) -> None:
        """Sets size equal to value.
        Precondition: Value is a list of tuples"""
        assert isinstance(value, list), "Value is not a list!"
        assert isinstance(value[0], tuple), "Elements aren't tuples!"
        # TODO: assert values are floats for consistency
        self._size = value
    
    def __init__(self, size: list[tuple[float, float]]):
        self.setSize(size)
    
    def __str__(self):
        return " ".join(map(str, self._size))


class Ligand:
    """
    Represents a simple smell embedded in Q space of dimensionality Q.

    Attributes
    ----------

    _id : int
        Identifies the odor
    _loc : list[float]
        Coordinates in chemical space. Dim=Q
    _dim : int
        Dimension of ligand (Q)
    _conc : float
        Concentration in molar, e.g., 1.5e-5 (NOT 0)
    _aff : float
        Temporary affinity value for a specific receptor
    _eff : float
        Temporary efficacy value for a specific receptor [0..1]
    _occ : float
        Temporary partial occupancy value for a specific receptor [0..1]
    _affs : list[float]
        affs for all recepters
    _effs : list[float]
        effs for all recepters
    _odors2 : list[Ligand] nested for faster calcs

    """
    
    @property
    def id(self) -> int:
        """Returns the id."""
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Sets id equal to value.
        Precondition: Value is an int"""
        assert isinstance(value, int), "Value is not a int!"
        self._id = value
    
    @property
    def loc(self) -> list[float]:
        """Returns the loc."""
        return self._loc

    @loc.setter
    def loc(self, value: list[float]) -> None:
        """Sets loc equal to value.
        Precondition: Value is a List"""
        assert isinstance(value, list), "Value is not a List!"        
        self._loc = value
    
    @property
    def dim(self) -> int:
        """Returns the dim."""
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        """Sets dim equal to value.
        Precondition: Value is an int"""
        assert isinstance(value, int), "Value is not a int!"
        self._dim = value
    
    @property
    def conc(self) -> float:
        """Returns the conc."""
        return self._conc

    @conc.setter
    def conc(self, value: float) -> None:
        """Sets conc equal to value.
        Precondition: Value is a nonZero number"""
        assert isinstance(value, Real), "Value is not a number!"
        assert value != 0, "Conc can't be 0!"
        self._conc = float(value)
    
    # FIXME: There seems to be a type mismatch from the stated class type and the set value type!
    def setAff(self, value: list[float]) -> None:
        """Sets aff equal to value.
        Precondition: Value is a float"""
        assert type(value) == float, "Value is not a float!"
        self._aff = value
    
    # FIXME: Same issue as the above fixme (type mismatch from attribute to set value)
    def setEff(self, value: list[float]) -> None:
        """Sets eff equal to value.
        Precondition: Value is a float btwn 0..1"""
        assert type(value) == float, "Value is not a float!"
        assert value >= 0 and value <= 1, "Eff is not btwn 0 and 1"
        self._eff = value
    
    def setOcc(self, value: float) -> None:
        """Sets occ equal to value.
        Precondition: Value is an float btwn 0..1"""
        assert type(value) == float, "Value is not a float!"
        assert value >= 0.0 and value <= 1.0, "Occ is not btwn 0 and 1"
        self._occ = value

    def appendToAffs(self, value: float) -> None:
        """adds aff equal to value.
        Precondition: Value is a float"""
        assert type(value) == float, "Value is not a float!"
        self._affs.append(value)

    def appendToEffs(self, value: float) -> None:
        """adds eff equal to value.
        Precondition: Value is a float"""
        assert type(value) == float, "Value is not a float!"
        self._effs.append(value)

    def appendToOdors2(self, value: Ligand) -> None:
        """adds odor2 equal to value.
        Precondition: odor2 is type of Ligand"""
        assert type(value) == Ligand, "Value is not a Ligand!"
        self._odors2.append(value)

    def getOdors2(self) -> list[Ligand]:
        """Returns _odors2."""
        return self._odors2


    #Initializer
    def __init__(self, Id, loc, conc):
        """Initializes ligand"""
        self.id = Id
        self.loc = loc
        self.dim = len(loc)
        self.conc = conc
        self._aff = 0.0
        self._eff = 0.0
        self._occ = 0.0
        self._affs = []
        self._effs = []
        self._odors2 = []        

    def __str__(self):
        """Returns description of Ligand"""
        return f"ID: {self._id} Loc: {self._loc} Conc: {self._conc}"


class Odorscene:
    """Represents a list of ligands embedded in Q-Space. Can resemble an odorant, a complex odorant,
    or a mixture of odorants.
    
    Attributes
    _id : int
        identifies the odor
    _dim : int
        dimension of odorscene (same as dim of attached ligands)
    _odors : list[Ligand]
        List of ligands that define the odorscene
    """

    @property
    def id(self) -> int:
        """Returns the id."""
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Sets id equal to value.
        Precondition: Value is an int"""
        assert isinstance(value, int), "Value is not a int!"
        self._id = value

    @property
    def dim(self) -> int:
        """Returns the dim."""
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        """Sets dim equal to value.
        Precondition: Value is an int"""
        assert type(value) == int, "Value is not a int!"
        self._dim = value

    @property
    def odors(self) -> list[Ligand]:
        """Returns the odors."""
        return self._odors

    @odors.setter
    def odors(self, value: list[Ligand]) -> None:
        """Sets odors equal to value.
        Precondition: Value is a List and dim of elements are equal."""
        assert type(value) == list, "Value is not a List!"
        self._odors = []
        # TODO: simplify
        i = 0
        while i < len(value):
            assert value[0].dim == value[i].dim, "Odors aren't all the same dim."
            self._odors.append(value[i])
            i += 1

    def __init__(self, Id, odors: list[Ligand]):
        """Initializes odorscene"""
        self.id = Id
        self.dim = odors[0].dim
        self.odors = odors
        
    def __str__(self):
        """Returns description of Odor"""
        # st = ""
        # for odor in self._odors:
        #     st = st + str(odor) + '\n'
        # return "ID: " + str(self._id) + '\n' + "Odors: \n" + st
        # Hotfix until Python 3.12
        n_ = '\n'
        return f"ID: {self.id}\nOdors: \n{n_.join(map(str, self._odors))}"


class Receptor:
    """Represents an odor receptor with center (x,y,z...) and radius of 
    sensitivity r.

    Attributes
    ----------
    _id : int
        identifies the receptor
    _mean : list
        list of means for affinity and efficacy Gaussian distributions. Length = Q
    _sdA : list
        List of standard deviations for affinity. Length = Q
    _sdE : list
        List of standard deviations for efficacy. Length = Q
    _covA : list
        sdA ^^ 2
    _covE : list
        sdE ^^ 2
    _scale : float
        A heuristic scalar value for the "strongest available affinity"
    _effScale : float
        A heuristic scalar value for the "strongest available affinity"
    _activ : float
        Total activation level of receptor
    _occ : float
        Total occupancy of receptor
    _affs : np.ndarray
        aff values of all odors for this receptor
    _effs : np.ndarray
        eff values of all odors for this receptor

    """
    
#Getters and Setters
    @property
    def id(self):
        """Returns id of receptor."""
        return self._id

    @id.setter
    def id(self, value):
        """Sets id to value
        Precondtion: value is an int"""
        assert type(value) == int, "Value is not a int!"
        self._id = value
    
    @property
    def mean(self):
        """Returns mean of receptor."""
        return self._mean

    @mean.setter
    def mean(self, value):
        """Sets id to value
        Precondtion: value is an list"""
        assert type(value) == list, "Value is not a float!"
        self._mean = value
    
    @property
    def sdA(self):
        """Returns the standard deviations for Affinity."""
        return self._sdA

    @sdA.setter
    def setSdA(self, value):
        """Sets sdA equal to value.
        Precondition: Value is a List with dim Q"""
        assert type(value) == list, "Value is not a List!"
        assert len(value) == len(self._mean), "Dimension is not consistent with dim of mean"
        self._sdA = value
    
    @property
    def sdE(self):
        """Returns the standard deviations for Efficacy."""
        return self._sdE

    @sdE.setter
    def sdE(self, value):
        """Sets sdE equal to value.
        Precondition: Value is a List with dim Q"""
        assert type(value) == list, "Value is not a List!"
        assert len(value) == len(self._mean), "Dimension is not consistent with dim of mean"
        self._sdE = value
    
    @property
    def covA(self):
        """Returns the covariance for affinity"""
        return self._covA
    
    # FIXME: This needs a little work to properly be converted into a property getter/setter
    @covA.setter
    def covA(self, value):
        """Sets covariance of affinity by squaring the sd."""
        covA = []
        i = 0
        while i < len(self._sdA):
            covA.append(self._sdA[i]**2)
            i += 1
        self._covA = covA
    
    @property
    def covE(self):
        """Returns the covariance for Efficacy"""
        return self._covE
    
    @covE.setter
    def covE(self, value):
        """Sets covariance of Efficacy by squaring the sd."""
        covE = []
        i = 0
        while i < len(self._sdE):
            covE.append(self._sdE[i]**2)
            i += 1
        self._covE = covE
    
    @property
    def scale(self):
        """Returns scale of receptor."""
        return self._scale

    @scale.setter
    def scale(self):
        """Sets scale based on mean"""
        self._scale = mvn.pdf(self.mean, self.mean, self.covA)

    @property
    def effScale(self):
        """Returns eff scale of receptor."""
        return self._effScale

    @effScale.setter
    def effScale(self):
        """Sets eff scale based on mean"""
        self._effScale = float(mvn.pdf(self.mean, self.mean, self.covE))

    @property
    def activ(self):
        return self._activ

    @activ.setter
    def activ(self, value):
        """Sets activation level for receptor"""
        self._activ = value
    
    def setOcc(self, value):
        """Sets occupancy for receptor"""
        self._occ = value
    
    def setOdoAmt(self,value):
        """sets amount of ligands 'adjacent' (<2SD) to receptor"""
        self._odoAmt = value

    @property
    def affs(self):
        """Returns affs for all ordors"""
        return self._affs

    @affs.setter
    def affs(self, value):
        """Sets affs for all ordors"""
        self._affs = value

    @property
    def effs(self):
        """Returns effs for all ordors"""
        return self._effs

    @effs.setter
    def effs(self, value):
        """Sets effs for all ordors"""
        self._effs = value


#Initializer
    def __init__(self, Id, mean, sda, sde):
        """Initializes a receptor."""
        self.id = Id
        self.mean = mean
        self.sdA = sda
        self.sdE = sde
        # FIXME: Hotfix until getters and setters are fixed for covA, covE, scale, and effScale (are autoassigned by setter)
        # Sol. is probably to change getter to calculate the values automagically
        self.covA = None
        self.covE = None
        self.scale = None
        self.activ = 0
        self.setOcc(0)
        self.setOdoAmt(0)
        self.effScale = None


    def __str__(self):
        """Returns receptor description"""
        # st = ""
        # for num in self.getMean():
        #     st = st + str(num) + ", "
        return f"ID {self.id} Mean: {', '.join(map(str, self.mean))}."
        # return "ID " + str(self._id) + " Mean: " + st[:-2] + "."  #Can add mean if prefer
    

class Epithelium:
    """Represents a list of receptors.
    Instance Attributes:
    _recs : list[Receptor]
    """
    
    @property
    def recs(self):
        """Returns the receptors."""
        return self._recs

    @recs.setter
    def recs(self, value):
        """Sets receptors equal to value.
        Precondition: Value is a List"""
        assert isinstance(value, list), "Value is not a List!"
        self._recs = value
    
    def __init__(self, recs):
        """Initializes a epithelium."""
        self.recs = recs

    def __str__(self):
        """Returns epithelium description"""
        n_ = '\n'
        return f"Epithelium contains the following receptors: \n{n_.join(map(str, self.recs))}"

class Text:
    """Holding experimental text to later store in text file"""
    
    def __init__(self, st, name):
        self._st = st
        self._name = name
        self._st2 = ""

######Functions for objects
def createLigand(dim: int, conc: int, qspace: QSpace, ID=0):
    """Returns an ligand with randomly (uniformly) generated ligand point coordinates
    in Q space."""
    i = 0
    loc = []
    while (i<dim):
        loc.append(random.uniform(-1000, 1000))
        i += 1
    return Ligand(ID, loc, conc)


def createOdorscene(dim: int, conc: list[float], amt: list[int], qspace: QSpace, Id = 0):
    """Returns an odorscene object with a certain amount of randomly generated ligands.
    Conc is a list of concentrations and amt is the amt of each conc (matched by index).
    Ex: conc = [1e-5, 5e-6, 1e-6] and amt = [6, 4, 2] This means:
    six randomly generated ligands at 1x10e-5 molar, four at 5x10e-6 molar,
    and two at 1x10e-6 molar.
    qspace is a qspace object.
    Precondition: Conc and amt are lists of equal length"""
    assert len(conc) == len(amt), "conc and amt are not lists of equal length"
    odors = []
    ID = 0
    i = 0
    while i < len(amt):
        ind = 0
        while ind < amt[i]:
            odor = createLigand(dim, conc[i], qspace, ID)
            odor = modifyLoc(odor, qspace, dim) 
            odors.append(odor)
            ind += 1
            ID += 1
        i += 1
    return Odorscene(Id, odors)

    
def modifyLoc(odorant: Ligand, qspace: QSpace, dim: int):
    """Modifies an odorant's location to be within the given qspace of the odorscene
    Precondition: QSpace dimensions are consistent with dim"""
    assert len(qspace.getSize()) == dim, "QSpace dimensions are not consistent with ligand locations"
    i = 0
    loc = odorant.loc
    while i < dim:
        loc[i] = ((loc[i] + (abs(qspace.getSize()[i][0]))) % (abs(qspace.getSize()[i][0]) +
                                abs(qspace.getSize()[i][1]))) + -1 * abs(qspace.getSize()[i][0])
        i += 1
    odorant.loc = loc
    return odorant


def createEpithelium(n: int, dim: int, qspace: QSpace, scale: list[float]=[.5,1.5], scaleEff=[], constMean=False):
    """Returns an epithelium with n receptors by calling createReceptor n times.
    SD of each receptor is a uniformly chosen # btwn scale[0] and scale[1]
    Precondition: n is an int"""
    assert type(n) == int, "n is not an integer"
    i = 0
    recs= []
    while i < n:
        recs.append(createReceptor(dim, qspace, scale, scaleEff, constMean, i))
        i += 1
    return Epithelium(recs)

def createReceptor(dim: int, qspace: QSpace, scale: list[float] = [0.5,1.5], scaleEff=[], constMean=False, Id=0):
    """Creates a receptor using meanType sdAtype and sdEtype as descriptions
    to how to randomly distribute those values.
    scaleEff is empty unless want to have a diff sd for aff and eff.
    Precondition: qspace must have "dim" dimensions"""
    assert len(qspace._size) == dim, "Qspace doesn't have right dimensions"
    mean = _distributeMean(dim, qspace, constMean)
    sdA = _distributeSD(dim, scale, [])
    sdE = _distributeSD(dim, scale, scaleEff)
    return Receptor(Id, mean, sdA, sdE)

def _distributeMean(dim: int, qspace: QSpace, constMean: bool):
    """Returns a list of means randomly distributed within the qspace based on the Type"""
    mean = []
    i = 0
    if constMean:
        while i < dim:
            mean.append(qspace.getSize()[i][1]/2.0)
            i+=1
    else:
        while i < dim:
            if params.DIST_TYPE_UNIF:
                mean.append(random.uniform(qspace.getSize()[i][0], qspace.getSize()[i][1]))
            elif params.DIST_TYPE_GAUSS:
                while True:
                    g = random.gauss(params.MU, params.SIG)
                    #if ((i==0 and g <= qspace.getSize()[i][1] and g >= 0) or (i==1 and g <= qspace.getSize()[i][1]) and g >= 0):
                    if (g <= qspace.getSize()[i][1] and g >= 0):
                        mean.append(g)
                        break
                #mean.append(random.gauss(params.MU, params.SIG))

            i += 1
    return mean

def _distributeSD(dim, scale, scaleEff):
    """Returns a list of standard deviations between scale[0] and scale[1] randomly distributed based on the Type
    Precondition: scale is a 2d list with #'s>=0"""
    assert scale[0] > 0, "scale is not a valid list"
    sd = []
    i = 0
    if len(scaleEff) == 0: #Want sd for aff and eff to be the same
        while i < dim:
            sd.append(random.uniform(scale[0],scale[1]))
            i += 1
    else:
        while i < dim:
            sd.append(random.uniform(scaleEff[0],scaleEff[1]))
            i += 1
    return sd


######## Activating Receptors/corresponding GL

def ActivateGL_QSpace(epith, odorscene, gl, fixed=True, c=1, sel="avg"):
    """Given an epithelium, odorscene, and Glomerular Layer, the GL is activated based on
    its corresponding 1:1 receptors. Returns string of data about odorscene and epithelium.
    If c!=1 (convergence ratio of glom:rec, then use updated function to calc non 1:1 glom:rec activation.
    Precondition: Epith and odorscene have the same dimension Q and # of receptors = len(gl)"""    
    assert len(epith.getRecs()[0].getMean()) == odorscene.getOdors()[0].getDim(), "Dimensions aren't equal"
    assert len(epith.getRecs()) == len(gl), "Receptors:GL is not 1:1"
    
    layers.clearGLactiv(gl)
    
    #Loop through each receptor and eventually calculate activation level
    for rec in epith.getRecs():
        #Set everything to 0
        activ = 0.0
        odors = []
        df = 0
        
        effScale = float(mvn.pdf(rec.getMean(), rec.getMean(), rec.getCovE())  )
        
        for odor in odorscene.getOdors():
            #First odorscene
            aff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovA())
            aff = aff / rec.getScale() #Scales it from 0 to 1
            #Now convert gaussian aff to kda
            
            aff = 10**((aff * (peak_affinity - minimum_affinity)) + minimum_affinity) ##peak_affinity etc. are global variables
            #print("aff is " + str(aff))
            odor.setAff(float(aff))
            
            if fixed:
                odor.setEff(1.0)
            else:
                eff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovE())
                eff = float(eff) / effScale #Scales it from 0 to 1
                odor.setEff(eff)
                #print("eff is " + str(eff))
            odors.append(odor)
            df += odor.getConc()/odor._aff
        
        i = 1
        for odor in odors:
            odor.setOcc( (1) / (1 + ( (odor._aff/odor.getConc()) * (1 + df - (odor.getConc() / odor._aff ) ) ) **m) ) #m=1
            activ += odor._eff * odor._occ
            
            #print("Occ is: " + str(odor._occ))
            #print(str(i) + " activation due to odorscene 1: " + str(activ_1))
            i += 1
        #print("activ level for rec " + str(activ))
        rec.setActiv(activ)
        gl[rec.getId()].setActiv(activ)
        
    if c != 1:
        glomRecConnNew(epith.getRecs(), gl, c)

#######Loading and Saving objecsts using CSV files

def saveLigand(odor, name):
    """Stores odor as one row in a CSV file with the following columns:
    A = ID# of ligand
    B = text label ('odorant membership')
    C = concentration of ligand in molar
    D...(Q-3) = the Q coordinates of the ligand point
    Precondtion: Name is a str"""
    assert type(name) == str, "name is not a string"
    st = "ID, Label, Conc "
    i = 0
    while i < odor.getDim():
        st = st + ", coord " + str(i)
        i += 1
    st = st + "\n" + str(odor.getId()) + ",' '," + str(odor.getConc())
    for loc in odor.getLoc():
        st = st + "," + str(loc)
    test = open(name + ".csv", "w")
    test.write(st)
    test.close

def saveOdorscene(odorScene, name):
    """Stores odor as a CSV file with the following format:
    First and second row: contains ID and dim of odorscene
    Every row after that symbolizes an odor with the following columns:
    A = ID# of ligand
    B = text label ('odorant membership')
    C = concentration of ligand in molar
    D...(Q-3) = the Q coordinates of the ligand point
    Precondtion: Name is a str"""
    assert type(name) == str, "name is not a string"
    st = "\n OdorSceneID, dim \n"
    st = st + str(odorScene.getId()) + "," + str(len(odorScene.getOdors())) + '\n'
    st = st + "ID, Label, Conc "
    
    i = 0
    while i < odorScene.getOdors()[0].getDim():
        st = st + ", coord " + str(i)
        i += 1
    for odor in odorScene.getOdors():
        st = st + "\n" + str(odor.getId()) + ",' '," + str(odor.getConc())
        for ii, loc in enumerate(odor.getLoc()):
            st = st + "," + str(loc)

    test = open(name + ".csv", "a")
    test.write(st)
    test.close

    """
    plt.plot(xaxis,yaxis, label="Odor Locations")
    plt.legend()
    plt.title("All Odors")
    plt.xlabel("x coordinates")
        
    plt.ylabel("y coordinates")
    #Set y_axis limit
    #axes = plt.gca()
    #axes.set_ylim([0,1.0]) #*****Change if using >30 recs
    
    pp = PdfPages('All Odors.pdf')
    pp.savefig()
    pp.close()
    if close == True: #No more data to add to the graph
        plt.close()
    """

def loadLigand(name, helper=False):
    """Returns an ligand from a CSV file with the given name.
    If helper is true, then it's being called from loadOdorscene and we
    don't want to skip the first line.
    Precondition: name exists and it's in CSV format AND the file is <= 2 lines"""
    assert type(name) == str, "name isn't a string"

    if helper == False:
        text = open(name)
        i = 0
        for l in text: #essentially just skip the first line and save the second
            if i == 1:
                line = l
            i += 1
    else:
        line = name
    comma1 = line.find(",")
    comma2 = line.find(",", comma1+1)
    comma3 = line.find(",", comma2+1)  #Comma before first loc coord
    commas = [line.find(",", comma3+1)]
    k = 0
    while commas[-1] != -1:
        commas.append(line.find(",", commas[k] + 1))
        k+=1
    ID = int(line[:comma1])
    conc = float(line[comma2+1:comma3])
    index = 0
    loc = [float(line[comma3+1:commas[index]])]
    while commas[index] != -1:
        loc.append(float(line[commas[index]+1:commas[index+1]])) #when commas[index+1]=-1 it cuts off the last digit
        index += 1
    loc[index] = float(str(loc[index]) + line[-1]) #Accounts for missing digit
    if helper == False:
        text.close()
    return Ligand(ID, loc, conc)


def loadOdorscene(name):
    """Returns an odorscene from a CSV file with the given name.
    Precondtion: name existsand it's in CSV format"""
    assert type(name) == str, "name isn't a string"
    text = open(name)
    i = 0
    odors = []
    for line in text:
        if i == 1:
            comma1 = line.find(",")
            Id = int(line[:comma1])
        if i > 2:
            odors.append(loadLigand(line, True))
        i += 1
    text.close()
    return Odorscene(Id, odors)

def saveReceptor(rec, name, helper=False):
    """Stores receptor as one row in a CSV file with the following columns:
    A = ID# of receptor
    B = text label ('receptor membership')
    C...X = list of mean
    X...Y = list of SD for affinity
    y...Z = list of SD for efficacy
    Precondtion: Name is a str"""
    assert type(name) == str, "name is not a string"
    dim = len(rec.getMean())
    i = 0
    st = ''
    m = ''
    a = ''
    e = ''
    mean = ''
    aff = ''
    eff = ''
    while i < dim:
        m = m + ", Mean " + str(i)
        a = a + ", Aff " + str(i)
        e = e + ", Eff " + str(i)
        mean = mean + "," + str(rec.getMean()[i])
        aff = aff + "," + str(rec.getSdA()[i])
        eff = eff + "," + str(rec.getSdE()[i])
        i += 1
    if helper == False:
        st = st + "ID, Label" + m + a + e + '\n'
    st = st + str(rec.getId()) + ",' '" + mean + aff + eff
    if helper:
        return st
    test = open(name + ".csv", "w")
    test.write(st)
    test.close

def saveEpithelium(epi, name):
    """Stores each receptor as one row in a CSV file with the following columns:
    A = ID# of receptor
    B = text label ('receptor membership')
    C...X = list of mean
    X...Y = list of SD for affinity
    y...Z = list of SD for efficacy
    Precondtion: Name is a str"""
    assert type(name) == str, "name is not a string"
    st = ''
    m = ''
    a = ''
    e = ''
    i = 0
    while i < len(epi.getRecs()[0].getMean()):
        m = m + ", Mean " + str(i)
        a = a + ", Aff " + str(i)
        e = e + ", Eff " + str(i)
        i += 1
    st = st + "ID, Label" + m + a + e + '\n'
    
    for rec in epi.getRecs():
        st = st + saveReceptor(rec, name, True) + '\n'
    test = open(name + ".csv", "w")
    test.write(st)
    test.close

def loadReceptor(name, helper=False):
    """Returns a receptor from a CSV file with the given name.
    If helper is true, then it's being called from loadEpithelium and some
    adjustments are made.
    Precondition: name exists and it's in CSV format AND the file is <= 2 lines"""
    assert type(name) == str, "name isn't a string"
    if helper == False:
        text = open(name)
        i = 0
        for l in text: #essentially just skip the first line and save the second
            if i == 1:
                line = l
            i += 1
    else:
        line = name
    comma1 = line.find(",")
    comma2 = line.find(",", comma1+1) #Comma before first mean coord
    commas = [line.find(",", comma2+1)]
    i = 0
    while commas[i] != -1:
        commas.append(line.find(",", commas[i]+1))
        i += 1
    dim = len(commas) / 3
    Id = int(line[:comma1])
    mean = [float(line[comma2+1:commas[0]])]
    index = 1
    while index < dim:
        mean.append(float(line[commas[index-1]+1:commas[index]]))
        index += 1
    aff = []
    while index < (2*dim):
        aff.append(float(line[commas[index-1]+1:commas[index]]))
        index += 1
    eff = []
    while index < (3*dim):
        eff.append(float(line[commas[index-1]+1:commas[index]])) #Last index of aff loses last digit due to -1
        index += 1
    eff[dim-1] = float(str(eff[dim-1]) + line[-1]) #Accounts for missing digit
    if helper == False:
        text.close()
    return Receptor(Id, mean, aff, eff)

def loadEpithelium(name):
    """Returns an epithelium from a CSV file with the given name.
    Precondition: name exists and it's in CSV format"""
    assert type(name) == str, "name isn't a string"
    recs = []
    text = open(name)
    i = 0
    for line in text:
        if i > 0:
            recs.append(loadReceptor(line, True))
        i += 1.
    text.close()
    return Epithelium(recs)

##### Making a list of sequentially different odorscenes
#createOdorscene(dim, conc, amt, qspace, Id = 0)
def sequentialOdorscenes(n, amt, dim, change, qspace):
    """Returns a list of n odorscenes, each one differs by change
    Amt=amount of ligand per odorscene
    warning: Doesn't call modify loc so loc could be out of qspace range"""
    odorscenes = []
    ligands = []
    
    #make amt ligands starting with [0,0,0...],[.1,.1,.1...]
    i = 0
    while i < amt:
        loc = []
        x = 0
        while x < dim:
            loc.append(i/10.0)
            x += 1
        ligands.append(Ligand(i, loc, 1e-5))
        i += 1
    odorscenes.append(Odorscene(0, ligands))
    
    #Creating rest of odorscenes
    i = 1    
    while i < n:
        ligands = []
        numOdors = 0
        odors = odorscenes[i-1].getOdors()
        while numOdors < amt:
            newOdorLoc = []
            for num in odors[numOdors].getLoc():
                newOdorLoc.append(num + change)
                
            ligands.append(Ligand(numOdors, newOdorLoc, 1e-5))
            numOdors += 1
        odorscenes.append(Odorscene(i, ligands))
        i += 1
    
    return odorscenes


####### Sum of Squares differentiation calculation - original
##Definitions: phi = receptor activation
##             dphi = difference in receptor activation due to two diff odorscenes
##             dpsi = difference in epithelium activation due to two diff odorscenes
## Maximum dpsi value = # of receptors in epithelium (if the first odorscene
## always activates the receptor = 1.0 and the other activates = 0.0)

def sumOfSquares(epithelium, odorscene, dn, fixed=False, c=1, gl=[]): 
    """Calculates differentiation between epithelium activation of odorscene before
    and after dn using sum of squares. Returns dpsi of the epithelium.
    If fixed=true, then efficacy will be fixed at 1 (only agonists)
    If c!=1, then use function to activate glom with 1:c ratio of Glom:Rec
    Precondtion: dn=list in correct dim"""
    assert odorscene.getDim()== len(dn), "dimension not consistent with dn"
    
    dPsi = 0
    recs2 = copy.deepcopy(epithelium.getRecs())
    layers.clearGLactiv(gl) #Sets gl activations and recConn back to 0.0
    
    counter = 0 #for storing info in rec2
    for rec in epithelium.getRecs():
        
        #Set everything to 0
        activ_1 = 0.0
        activ_2 = 0.0
        totOcc = 0.0
        odors = []
        odors2 = []
        rec._activ = 0.0
        rec._occ = 0.0
        rec._odoAmt = 0.0
        df = 0
        df2 = 0
        dphi = 0
        effScale = float(mvn.pdf(rec.getMean(), rec.getMean(), rec.getCovE())  ) 
        
        for odor in odorscene.getOdors():
            #First odorscene
            aff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovA())
            aff = aff / rec.getScale() #Scales it from 0 to 1
            #Now convert gaussian aff to kda
            aff = 10**((aff * (peak_affinity - minimum_affinity)) + minimum_affinity) ##peak_affinity etc. are global variables
            
            #print("aff is " + str(aff))
            odor.setAff(float(aff))
            if fixed:
                odor.setEff(1.0)
            else:
                eff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovE())
                eff = float(eff) / effScale #Scales it from 0 to 1
                odor.setEff(eff)
            odors.append(odor)
            df += odor.getConc()/odor._aff
                
            
            #Second Odorscene
            newLoc = []  #Calculating new location
            index = 0
            while index < len(dn):
                newLoc.append(odor.getLoc()[index] + dn[index])
                index += 1
            newOdor = Ligand(odor.getId(), newLoc, odor.getConc())
            
            aff2 = mvn.pdf(newLoc, rec.getMean(), rec.getCovA())
            aff2 = aff2 / rec.getScale() #Scales it from 0 to 1
            aff2 = 10**((aff2 * (peak_affinity - minimum_affinity)) + minimum_affinity)
            #print("aff2 is " + str(aff2))
            newOdor.setAff(float(aff2))
            if fixed:
                newOdor.setEff(1.0)
            else:
                eff2 = mvn.pdf(newLoc, rec.getMean(), rec.getCovE())
                eff2 = float(eff2) / effScale #Scales it from 0 to 1
                newOdor.setEff(eff2)
            odors2.append(newOdor)
            df2 += newOdor.getConc()/newOdor._aff
                
        i = 1
        for odor in odors:
            odor.setOcc( (1) / (1 + ( (odor._aff/odor.getConc()) * (1 + df - (odor.getConc() / odor._aff ) ) ) **m) ) #m=1
            activ_1 += odor._eff * odor._occ
            rec._occ += odor._occ #Solely for printing individual receptor activations in experiments
            rec._odoAmt += adjOdors(rec, odor)
            #print(str(i) + " activation due to odorscene 1: " + str(activ_1))
            i += 1
        i = 1
        
        rec.setActiv(activ_1) #Solely for printing individual receptor activations in experiments
        
        for odor2 in odors2:
            odor2.setOcc( (1) / (1 + ( (odor2._aff/odor2.getConc()) * (1 + df2 - (odor2.getConc() / odor2._aff ) ) ) **m) ) #m=1
            activ_2 += odor2._eff * odor2._occ
            #print(str(i) + " activation due to odorscene 2: " + str(activ_2))
            i += 1
        
        recs2[counter].setActiv(activ_2)
        
        #print("activ_1 is " + str(activ_1))
        #print("activ_2 is " + str(activ_2))
        dPhi = (activ_1 - activ_2) #########(Maximum value will be 1 or -1 = make sure this is true)
        dPsi += dPhi**2
        
        
        counter += 1
    
    #print("rec dPsi is: " + str(math.sqrt(dPsi))
    
    if c != 1:
        
        gl2 = copy.deepcopy(gl)
        
        conn = glomRecConnNew(epithelium.getRecs(), gl, c, [])
        glomRecConnNew(recs2, gl2, c, conn)

        count = 0
        dPsi = 0
        while count < len(gl):
            dPhi = (gl[count].getActiv() - gl2[count].getActiv())
            # print("gl 1 activ is " + str(gl[count].getActiv()))
            # print("gl 2 activ is " + str(gl2[count].getActiv()))
            dPsi += dPhi**2
            count += 1
        #print("dPsi is " + str(math.sqrt(dPsi)))
        return math.sqrt(dPsi)
    else:
        return math.sqrt(dPsi)




####### Sum of Squares differentiation calculation - vectorized
##Definitions: phi = receptor activation
##             dphi = difference in receptor activation due to two diff odorscenes
##             dpsi = difference in epithelium activation due to two diff odorscenes
## Maximum dpsi value = # of receptors in epithelium (if the first odorscene
## always activates the receptor = 1.0 and the other activates = 0.0)

def sumOfSquaresVectorized(epithelium, odorscene, dn, repIndex, fixed=False, c=1, gl=[]): 
    
    #print("CALLED AGAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    """Calculates differentiation between epithelium activation of odorscene before
    and after dn using sum of squares. Returns dpsi of the epithelium.
    If fixed=true, then efficacy will be fixed at 1 (only agonists)
    If c!=1, then use function to activate glom with 1:c ratio of Glom:Rec
    Precondtion: dn=list in correct dim"""
    
    #assert odorscene.getDim()== len(dn), "dimension not consistent with dn"
    
    dPsi = 0
    recs2 = copy.deepcopy(epithelium.getRecs())
    layers.clearGLactiv(gl) #Sets gl activations and recConn back to 0.0
    
    
    #print("before loop levl 1:" + str(time.time()))
    counter = 0 #for storing info in rec2
    for rec in epithelium.getRecs():
        
        #Set everything to 0
        activ_1 = 0.0
        activ_2 = 0.0
        totOcc = 0.0
        odors = []
        odors2 = []
        rec._activ = 0.0
        rec._occ = 0.0
        rec._odoAmt = 0.0
        df = 0
        df2 = 0
        dphi = 0
        
        #print("getMean=" + str(rec.getMean()) + "TIME:"+ str(time.time()))
        #print("getCovE=" + str(rec.getCovE()) + "TIME:"+ str(time.time()))
        
        #effScale = float(mvn.pdf(rec.getMean(), rec.getMean(), rec.getCovE())  )
        effScale = rec.getEffScale()
        #print("effScal=" + str(effScale) + "TIME:"+ str(time.time()))
        
        '''
        affs = rec.getAffs()
        effs = rec.getEffs()
        '''
        oi = 0
        for odor in odorscene.getOdors():
            #First odorscene
            startTime = time.time()

            odor.setAff(odor._affs[counter])
            odor.setEff(odor._effs[counter])
            
            
            newOdors = odor.getOdors2()
            odor2 = newOdors[repIndex]
                

            odor2.setAff(odor2._affs[counter])
            odor2.setEff(odor2._effs[counter])
                
            #df += prepareOdor(odor, rec, fixed, odors, effScale) 
            odors.append(odor)
            df += odor.getConc()/odor._aff
            #print("time elapsed prepareOdor:"+ str((time.time() - startTime)))
            
            #Second Odorscene
            '''
            newLoc = []  #Calculating new location
            #index = 0
            #while index < len(dn):
            for index, dnItem in enumerate(dn):    
                #newLoc.append(odor.getLoc()[index] + dn[index])
                newLoc.append(odor.getLoc()[index] + dnItem)
                #index += 1
            newOdor = Ligand(odor.getId(), newLoc, odor.getConc())
            
            startTime = time.time()
            '''
                
            odors2.append(odor2)
            df2 += odor2.getConc()/odor2._aff
            
            #df2 += prepareOdor(newOdor, rec, fixed, odors2, effScale) 
                
            oi+=1
            
        #i = 1
        for odor in odors:
        #for odor in odorscene.getOdors():    
            #startTime = time.time()
            odor.setOcc( (1) / (1 + ( (odor._aff/odor.getConc()) * (1 + df - (odor.getConc() / odor._aff ) ) ) **m) ) #m=1
            # print("time elapsed odor setOcc:"+ str((time.time() - startTime)))
            activ_1 += odor._eff * odor._occ
            rec._occ += odor._occ #Solely for printing individual receptor activations in experiments
            #startTime = time.time()
            rec._odoAmt += adjOdors(rec, odor)
            #print("time elapsed adjOdors:"+ str((time.time() - startTime)))
            #print(str(i) + " activation due to odorscene 1: " + str(activ_1))
            #i += 1
        #i = 1
        
        rec.setActiv(activ_1) #Solely for printing individual receptor activations in experiments
        
        for odor2 in odors2:
            #startTime = time.time()
            odor2.setOcc( (1) / (1 + ( (odor2._aff/odor2.getConc()) * (1 + df2 - (odor2.getConc() / odor2._aff ) ) ) **m) ) #m=1
            #print("time elapsed odor2 setOcc:"+ str((time.time() - startTime)))
            activ_2 += odor2._eff * odor2._occ
            #print(str(i) + " activation due to odorscene 2: " + str(activ_2))
            #i += 1
        
        recs2[counter].setActiv(activ_2)
        
        #print("activ_1 is " + str(activ_1))
        #print("activ_2 is " + str(activ_2))
        dPhi = (activ_1 - activ_2) #########(Maximum value will be 1 or -1 = make sure this is true)
        dPsi += dPhi**2
        
        
        counter += 1
        #print("counter is:" + str(counter) + ":::"+ str(time.time()))
    
    #print("after loop levl 1:" + str(time.time()))
    #print("rec dPsi is: " + str(math.sqrt(dPsi)))
    
    if c != 1:
        
        gl2 = copy.deepcopy(gl)
        
        conn = glomRecConnNew(epithelium.getRecs(), gl, c, [])
        glomRecConnNew(recs2, gl2, c, conn)

        #count = 0
        dPsi = 0
        for count, glItem in enumerate(gl):
        #while count < len(gl):
            #dPhi = (gl[count].getActiv() - gl2[count].getActiv())
            dPhi = (glItem.getActiv() - gl2[count].getActiv())

            #print("gl 1 activ is " + str(gl[count].getActiv()))
            #print("gl 2 activ is " + str(gl2[count].getActiv()))
            dPsi += dPhi**2
            #count += 1
        #print("dPsi is " + str(math.sqrt(dPsi)))
        #print("END OF sumOfSquares:" + str(time.time()))
        #return math.sqrt(dPsi)
    #else:
    #print("END OF sumOfSquares:" + str(time.time()))
    return math.sqrt(dPsi)


def prepareOdor(odor, rec, fixed, odors, effScale): 
    
    aff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovA())
    #print("time elapsed odor pdf:"+ str((time.time() - startTime)))
    aff = aff / rec.getScale() #Scales it from 0 to 1
    #Now convert gaussian aff to kda
    #startTime = time.time()
    aff = 10**((aff * (peak_affinity - minimum_affinity)) + minimum_affinity) ##peak_affinity etc. are global variables
    #print("time elapsed odor aff:"+ str((time.time() - startTime)))        
    #print("aff is " + str(aff))
    odor.setAff(float(aff))
    if fixed:
        odor.setEff(1.0)
    else:
        eff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovE())
        eff = float(eff) / effScale #Scales it from 0 to 1
        odor.setEff(eff)
    odors.append(odor)
    return odor.getConc()/odor._aff





def adjOdors(rec, odor):
    """Returns 1 if odor is within 2 SD of the rec mean. Otherwise returns 0"""
    #rec._sdA and rec._mean and odor.getLoc()
    #First find avg of sdA
    i = 0
    avg = 0
    dim = odor.getDim()
    while i < dim:
        avg += rec._sdA[i]
        i += 1
    avg = float(avg)/float(dim)
    #Find Euc distance
    index = 0
    num = 0.0
    while index < dim:
        num += (float(rec._mean[index]) - float(odor.getLoc()[index]))**2
        index += 1
    num = math.sqrt(num)
    #print("euc dist is " + str(num))
    #print("avg aff is " + str(avg))
    if num <= (2.0*avg):
        return 1
    else:
        return 0


def sumOfSquares2(epithelium, odorscene1, odorscene2, fixed=False):
    """Calculates dPsi for two given odorscenes (instead of one odorscene and dn).
    Precondtion: odorscene1 and odorscene 2 have the same dim and same # of odors"""""
    #assert odorscene1.getDim()==odorscene2.getDim(), "ligand dimensions are different"
    #assert len(odorscene1.getOdors()) == len(odorscene2.getOdors()), "two odorscenes don't have the same number of ligands to compare"
    peak_affinity = -8     # literally 10e-8, not influenced by minimum_affinity value
    minimum_affinity = 2   # asymptotic affinity exponent, negligible
    dPsi = 0
    for rec in epithelium.getRecs():
        #Set everything to 0
        activ_1 = 0.0
        activ_2 = 0.0
        odors = []
        odors2 = []
        df = 0
        df2 = 0
        dphi = 0
        effScale = float(mvn.pdf(rec.getMean(), rec.getMean(), rec.getCovE())  ) 
        
        i = 0
        while i < len(odorscene1.getOdors()):
            #First odorscene
            odor = odorscene1.getOdors()[i]
            aff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovA())
            aff = aff / rec.getScale() #Scales it from 0 to 1
            #if aff > 1e-128:   #Need so don't divide by 0 when calc df
            #Now convert gaussian aff to kda
            aff = 10**((aff * (peak_affinity - minimum_affinity)) + minimum_affinity)
            odor.setAff(float(aff))
            if fixed:
                odor.setEff(1.0)
            else:
                eff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovE())
                eff = float(eff) / effScale #Scales it from 0 to 1
                odor.setEff(eff)
            odors.append(odor)
            df += odor.getConc()/odor._aff
              
            odor2 = odorscene2.getOdors()[i]
            aff2 = mvn.pdf(odor2.getLoc(), rec.getMean(), rec.getCovA())
            aff2 = aff2 / rec.getScale() #Scales it from 0 to 1
            #if aff2 > 1e-128:   #Need so don't divide by 0 when calc df
            #Now convert gaussian aff to kda
            aff2 = 10**((aff2 * (peak_affinity - minimum_affinity)) + minimum_affinity)
            odor2.setAff(float(aff2))
            if fixed:
                odor2.setEff(1.0)
            else:
                eff2 = mvn.pdf(odor2.getLoc(), rec.getMean(), rec.getCovE())
                eff2 = float(eff2) / effScale #Scales it from 0 to 1
                odor2.setEff(eff2)
            odors2.append(odor2)
            df2 += odor2.getConc()/odor2._aff
            i += 1
            
        i = 0
        while i < len(odors):
            odor = odors[i]
            odor.setOcc( (1) / (1 + ( (odor._aff/odor.getConc()) * (1 + df - (odor.getConc() / odor._aff ) ) ) **m) ) #m=1
            activ_1 += odor._eff * odor._occ
                
            odor2 = odors2[i]
            odor2.setOcc( (1) / (1 + ( (odor2._aff/odor2.getConc()) * (1 + df2 - (odor2.getConc() / odor2._aff ) ) ) **m) ) #m=1
            activ_2 += odor2._eff * odor2._occ                
            i+=1
                
        dPhi = (activ_1 - activ_2) #########(Maximum value will be 1 or -1 = make sure this is true)
        dPsi += dPhi**2
    return math.sqrt(dPsi)


def dPsiBarCalcDiag(epithelium, odorscene, r, fixed=False):
    """Calculates dPsiBar = the average dPsi value of an odorscene that
    changes location by the same amplitude r but different directions based on
    diagnols.
    fixed is fixing efficacy=1"""
    totalDpsi = 0
    #calc new dn (for now just do 4 diagnals regardless of dim)
    dim = odorscene.getDim()
    dn = []
    dn2 = []
    dn3 = []
    dn4 = []
    i=0
    sign = 1
    while i < dim:
        dn.append(r*(1/math.sqrt(dim)))
        dn2.append(-1*r*(1/math.sqrt(dim)))
        dn3.append(sign*r*(1/math.sqrt(dim)))
        dn4.append(-1*sign*r*(1/math.sqrt(dim)))
        sign = -1*sign
        i += 1
    #calc all dPsi values
    totalDpsi += sumOfSquares(epithelium, odorscene, dn, fixed) + sumOfSquares(epithelium, odorscene, dn2, fixed)
    totalDpsi += sumOfSquares(epithelium, odorscene, dn3, fixed) + sumOfSquares(epithelium, odorscene, dn4, fixed)
    
    return totalDpsi/4.0

def dPsiBarCalcAnglesOrig(epithelium, odorscene, r, fixed=False, text=None, c=1, gl=[]):
    """Calculates dPsiBar = the average dPsi value of an odorscene that
    changes location by the same amplitude r but "rep" different directions based on
    randomized angles."""
    
    rep = 10.0
    amtOfDir = 0
    totalDpsi = 0
    totalAct = [] #Solely for recording rec activ in text file
    totalOcc = [] #Solely for recording rec occupance in text file
    dim = odorscene.getDim()
    while amtOfDir < rep:
        #Create randomized list of angles
        angles = []
        for i in range(dim-1):
            if i == dim-2: #if last angle
                angles.append(random.uniform(0,(2*math.pi)))
            else:
                angles.append(random.uniform(0, math.pi))
        #Create dn = amount of change (length of line in each dim given vector r)
        dn = []
        for i in range(dim):
            dn.append(r)
            if i == dim-1: #if last angle
                for angle in angles:
                    dn[dim-1] *= math.sin(angle)
            else:
                j=0
                while j < i:
                    dn[i] *= math.sin(angles[j])
                    j+=1
                dn[i] *= math.cos(angles[i])
        totalDpsi += sumOfSquares(epithelium, odorscene, dn, fixed, c, gl)
        amtOfDir += 1

    if text != None:
        recToText(epithelium, gl, c, text)
    return totalDpsi/rep



def dPsiBarCalcAngles(epithelium, odorscene, r, fixed=False, text=None, c=1, gl=[]):
    """Calculates dPsiBar = the average dPsi value of an odorscene that
    changes location by the same amplitude r but "rep" different directions based on
    randomized angles."""

    #print("start of dPsiBarCalcAngles")

    
    #rep = 10.0
    rep = ANGLES_REP
    amtOfDir = 0
    totalDpsi = 0
    totalAct = [] #Solely for recording rec activ in text file
    totalOcc = [] #Solely for recording rec occupance in text file
    dim = odorscene.getDim()
    
    while amtOfDir < rep:
        '''
        startTime = time.time()
        print("start of a rep:" + str(startTime))

        #Create randomized list of angles
        angles = []
        for i in range(dim-1):
            #print("start of first for:" + str(i) +":"+ str(time.time()))

            if i == dim-2: #if last angle
                angles.append(random.uniform(0,(2*math.pi)))
            else:
                angles.append(random.uniform(0, math.pi))
        #Create dn = amount of change (length of line in each dim given vector r)
            #print("end of first for:" + str(i) +":"+ str(time.time()))

        dn = []
        for i in range(dim):
            #print("start of 2nd for:" + str(i) +":"+ str(time.time()))

            dn.append(r)
            if i == dim-1: #if last angle
                for angle in angles:
                    dn[dim-1] *= math.sin(angle)
            else:
                j=0
                while j < i:
                    dn[i] *= math.sin(angles[j])
                    j+=1
                dn[i] *= math.cos(angles[i])
                
            #print("end of 2nd for:" + str(i) +":"+ str(time.time()))
        '''    
        dn = []

        #print("bfore sumOfSquares:" +str(amtOfDir) + ":"+ str(time.time()))
        totalDpsi += sumOfSquaresVectorized(epithelium, odorscene, dn, amtOfDir, fixed, c, gl)
        #print("after sumOfSquares:" +str(amtOfDir) + ":"+ str(time.time()))
       
        amtOfDir += 1
        
        #print("end of a rep:" + str(time.time()))
        #print("time elapsed a rep:" + str(time.time() - startTime))
        
    if text != None:
        recToText(epithelium, gl, c, text)
    return totalDpsi/rep
    
def recToText(epithelium, gl, c, text):
    """Stores rec activ and rec occ from epi into a text obj"""
    if c != 1:
        num = convStrToNum(text._st[-4:])
    if text._name == "exp1":
        for rec in epithelium.getRecs():
            text._st += "," + str(rec._activ)
        for rec in epithelium.getRecs():
            text._st += "," + str(rec._occ)
        for rec in epithelium.getRecs():
            text._st += "," + str(rec._odoAmt)
        text._st += '\n'
    elif text._name == "exp2":
        n = 0
        for rec in epithelium.getRecs():
            text._st += "Rec" + str(n) + "," + str(rec._activ) + "," + str(rec._occ) + "," + str(rec._odoAmt) + '\n'
            n += 1
        text._st += '\n' #extra space
    if c!= 1:
        text._st2 += "glom_numOdo=" + str(num)
        for glom in gl:
            text._st2 += "," + str(glom._activ)
        text._st2 += '\n'
        
def convStrToNum(s):
    """Given string s with either 3, 2 or 1 num at the end, converts that num to a int"""
    try:
        num = int(s[-3:])
    except:
        try:
            num = int(s[-2:])
        except:
            num = int(s[-1])
    return num


def colorMapSumOfSquares(epithelium, odorscenes, r, qspace):
    """Creates a colorMap with Q-Space as the x and y axis and dPsi_bar as the
    color for each ligand. dPsi_bar = avg differentiation that occurs in each
    point in Q-Space given many small changes in the odor at that loc.
    Preconditions: Odorscenes is a list of odorscenes containing one ligand. All the ligands
    fill up Qspace. dim = 2d
    WARNING: only works in 2D"""
    assert odorscenes[0].getDim()==2, "dimension must be 2D!"
    ##Create graph of all 0's
    graph = []
    maxX = qspace.getSize()[0][1]
    maxY = qspace.getSize()[1][1]
    x = 0 
    y = 0
    while y < maxY * params.PIXEL_PER_Q_UNIT:
        row = []
        x = 0
        while x < maxX * params.PIXEL_PER_Q_UNIT:
            row.append(0)
            x += 1
        graph.append(row)
        y += 1
    print(graph)
    print("before \n")
    print(len(odorscenes))
    ##Similar code to above - calc individual dPsi for ligands
    for odorscene in odorscenes:

        # eff != 1
        # dPsiBar = dPsiBarCalcAnglesOrig(epithelium, odorscene, r)  ####Or use diaganols
        
        # eff = 1
        dPsiBar = dPsiBarCalcAnglesOrig(epithelium, odorscene, r, True) 
        print(odorscene.getOdors()[0].getLoc())

        graph[int(params.PIXEL_PER_Q_UNIT*(odorscene.getOdors()[0].getLoc()[1]))][int(params.PIXEL_PER_Q_UNIT*(odorscene.getOdors()[0].getLoc()[0]))] = dPsiBar
    
    print("-----------------------------")
    print(graph)


    #     #TESTING FOR RECEPTOR ELLIPSE ADD-ON
    ells_sda = []
    ells_sde = []
    ii = -1
    for i, rec in enumerate(epithelium.getRecs()): 
            if params.RECEPTOR_INDEX == 'ALL':
                ii=i
            else:
                ii= params.RECEPTOR_INDEX      
            if i == ii:   
                print("rec" + str(rec.getId()) + ":")
                print("Mean:" + str(rec.getMean()))
                print("SdA:" + str(rec.getSdA()))
                # print("eff: ")
                # for e in rec.getEffs():
                #     print(e)
                # print("aff: " + str(rec.getAffs))
                print("Qspace size is " + str(qspace.getSize()[1][1]))
        
                qspaceBoundary = qspace.getSize()[1][1]
                # ang = rnd.rand()*360
                ang = rnd.rand()
                ang1 = rnd.rand()
                ang2 = rnd.rand()
                """
                ells.append(Ellipse(xy=rec.getMean(), width=rec.getSdA()[0]*standardDeviationNumber, height=rec.getSdE()[0]*standardDeviationNumber, angle=ang))
                ells.append(Ellipse(xy=rec.getMean(), width=rec.getSdA()[1]*standardDeviationNumber, height=rec.getSdE()[1]*standardDeviationNumber, angle=ang))

                """
                if params.SHOW_SDA_ELLIPSE:
                    # ells_sda.append(Ellipse(xy=rec.getMean(), width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang))
                    
                    ells_sda.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN, width=params.MOCK_RECEPTOR_SDA[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDA[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    ells_sda.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN1, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang1))
                    ells_sda.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN2, width=params.MOCK_RECEPTOR_SDA2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDA2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang2))
        

                    # ***************** UNCOMMENT HERE TO PLOT RECEPTORS IN TORUS **************************
                    # newMeanLeft = [rec.getMean()[0]-qspaceBoundary, rec.getMean()[1]]
                    # print("newMeanLeft = " + str(newMeanLeft))
                    # print("newMeanLeft x value is " + str(rec.getMean()[0] - qspaceBoundary) + "!!!!! mean is " + str(rec.getMean()) + "!!!!! qspace boundary is " + str(qspaceBoundary))
                    # ells_sda.append(Ellipse(xy=newMeanLeft, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    # newMeanRight = [rec.getMean()[0] + qspaceBoundary, rec.getMean()[1]]
                    # ells_sda.append(Ellipse(xy=newMeanRight, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    # newMeanBottom = [rec.getMean()[0], rec.getMean()[1]-qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMeanBottom, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    # newMeanTop = [rec.getMean()[0], rec.getMean()[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMeanTop, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    # # bottom left
                    # newMean1bottomLeft=[rec.getMean()[0]-qspaceBoundary, rec.getMean()[1]-qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMean1bottomLeft, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))            

                    # # bottom right
                    # newMean1bottomRight=[rec.getMean()[0] + qspaceBoundary, rec.getMean()[1]-qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMean1bottomRight, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))            
            
                    # # top left
                    # newMean1topLeft=[rec.getMean()[0]-qspaceBoundary, rec.getMean()[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMean1topLeft, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))

                    # # top right
                    # newMean1topRight=[rec.getMean()[0] + qspaceBoundary, rec.getMean()[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMean1topRight, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))

                    
                if params.SHOW_SDE_ELLIPSE:
                    # ells_sde.append(Ellipse(xy=rec.getMean(), width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    
                    ells_sde.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN, width=params.MOCK_RECEPTOR_SDE[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDE[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    ells_sde.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN1, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang1))
                    ells_sde.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN2, width=params.MOCK_RECEPTOR_SDE2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDE2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang2))

                    # ***************** UNCOMMENT HERE TO PLOT RECEPTORS IN TORUS **************************
                    # newMeanLeft = [rec.getMean()[0] - qspaceBoundary, rec.getMean()[1]]
                    # ells_sde.append(Ellipse(xy=newMeanLeft, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    # newMeanRight = [rec.getMean()[0] + qspaceBoundary, rec.getMean()[1]]
                    # ells_sde.append(Ellipse(xy=newMeanRight, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    # newMeanBottom = [rec.getMean()[0], rec.getMean()[1] - qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMeanBottom, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    # newMeanTop = [rec.getMean()[0], rec.getMean()[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMeanTop, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))
                    
                    # # bottom left
                    # newMean1bottomLeft=[rec.getMean()[0]-qspaceBoundary, rec.getMean()[1]-qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMean1bottomLeft, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))            

                    # # bottom right
                    # newMean1bottomRight=[rec.getMean()[0] + qspaceBoundary, rec.getMean()[1]-qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMean1bottomRight, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))            
            
                    # # top left
                    # newMean1topLeft=[rec.getMean()[0]-qspaceBoundary, rec.getMean()[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMean1topLeft, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))

                    # # top right
                    # newMean1topRight=[rec.getMean()[0] + qspaceBoundary, rec.getMean()[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMean1topRight, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang, zorder=10))


                #ells.append(Ellipse(xy=rnd.rand(2)*10, width=rnd.rand(), height=rnd.rand(), angle=rnd.rand()*360))
    
    fig = plt.figure()

    ax = fig.add_subplot(111, aspect='equal')
    ax.add_patch


    #ax2=fig.add_subplot(111, label="2", frame_on=False)


    if params.SHOW_SDA_ELLIPSE:
        for e in ells_sda:
            print("The center is " + str(e.center))
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rnd.rand())
            #e.set_facecolor(rnd.rand(3))
            #e.set_facecolor('red')
            e.set_edgecolor(params.SDA_COLOR)
            e.set_fill(params.SDA_FILL)
            if params.SDA_FILL:
                e.set_facecolor(params.SDA_COLOR)
            e.set_label("SDA")
            e.set_linewidth (params.LINE_WIDTH)
            
    
    if params.SHOW_SDE_ELLIPSE:
        for e in ells_sde:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rnd.rand())
            #e.set_facecolor(rnd.rand(3))
            #e.set_facecolor('blue')
            e.set_edgecolor(params.SDE_COLOR)
            if params.SDE_FILL:
                e.set_facecolor(params.SDE_COLOR)
            e.set_fill(params.SDE_FILL)
            e.set_label("SDE")
            e.set_linewidth (params.LINE_WIDTH)

        print(qspace.getSize()[0])
        ax.set_xlim(qspace.getSize()[0])
        ax.set_ylim(qspace.getSize()[0])


    # PLOT ODORS
        #draw odor locations for specified odorscene
    #     locXaxis = []
    #     locYaxis = []
    #     for odorscene in odorscenes:
    #         for odor in odorscene.getOdors():
    #             print(odor.getLoc())
    #             locXaxis.append(odor.getLoc()[0])
    #             locYaxis.append(odor.getLoc()[1])

    # plt.scatter(locXaxis,locYaxis, s=4, c='black')

    # #END TESTING
 



    #ColorMap
    #matplotlib.pylab.matshow(graph, fignum="Research", cmap=matplotlib.pylab.cm.Greys) #Black = fully active
    #matplotlib.pylab.matshow(graph, fignum="Research", cmap=matplotlib.pylab.cm.YlOrRd) #Black = fully active

    im = matplotlib.pylab.imshow(graph, cmap=matplotlib.pylab.cm.YlOrRd, interpolation="nearest", vmin=0, vmax=1, origin="lower", extent=[0,4,0,4]) #Black = fully active

    plt.title("Differentiation in QSpace")
    plt.xlabel("X")
    plt.ylabel("Y")
    fig.colorbar(im)
    pp = PdfPages('ColorMap' + str(qspace.getSize()[0]) + '.pdf')
        
    pp.savefig()
    pp.close()

    #Not closing it will add odor locations to it
    plt.close()

def dPsiBarCalcDns(odorscene, r, rep):
    """Calculates dPsiBar = the average dPsi value of an odorscene that
    changes location by the same amplitude r but "rep" different directions based on
    randomized angles."""

    #print("start of dPsiBarCalcDns")

    
    #rep = 10.0
    #rep = 2.0
    amtOfDir = 0
    #totalDpsi = 0
    #totalAct = [] #Solely for recording rec activ in text file
    #totalOcc = [] #Solely for recording rec occupance in text file
    dim = odorscene.getDim()
    
    while amtOfDir < rep:
        
        #startTime = time.time()
        #print("start of a rep:" + str(startTime))

        #Create randomized list of angles
        angles = []
        for i in range(dim-1):
            #print("start of first for:" + str(i) +":"+ str(time.time()))

            if i == dim-2: #if last angle
                angles.append(random.uniform(0,(2*math.pi)))
            else:
                angles.append(random.uniform(0, math.pi))
        #Create dn = amount of change (length of line in each dim given vector r)
            #print("end of first for:" + str(i) +":"+ str(time.time()))

        dn = []
        for i in range(dim):
            #print("start of 2nd for:" + str(i) +":"+ str(time.time()))

            dn.append(r)
            if i == dim-1: #if last angle
                for angle in angles:
                    dn[dim-1] *= math.sin(angle)
            else:
                j=0
                while j < i:
                    dn[i] *= math.sin(angles[j])
                    j+=1
                dn[i] *= math.cos(angles[i])
                
            #print("end of 2nd for:" + str(i) +":"+ str(time.time()))
        
       
        amtOfDir += 1

    return dn

def dPsiBarSaturation(epithelium, r, qspace, pdfName, labelName, excelName, fixed=False, c=1, plotTitle="", close=False, purp='', graphIt=True):
    """Define x amount of odorscenes with one ligand per odorscene, then with two ligands...
    then calculate dPsibar for each group of odorscene and graph to find saturation at certain
    ligand number.
    if fixed=true than efficacy=1
    if close = True, then graph is closed after this round of data.
    precondition: c = integer, fixed and close = Boolean"""
    
    startTime = time.time()
    print("start of dPsiBarSaturation:" + str(startTime))

    size = ODOR_REPETITIONS #amount of odorscenes we want to avg out
    #conc = 1e-5
    conc = params.ODOR_CONCENTRATION
    gl = layers.createGL(len(epithelium.getRecs())) #Only if using newly modified gl:rec n:1 ratio
    
    
    #Instantiate odorscene and ligand lists
    '''
    odorscenes1 = []
    ligands2 = []
    odorscenes2 = []
    ligands3 = []
    odorscenes3 = []
    ligands4 = []
    odorscenes4 = []
    ligands5 = []
    odorscenes5 = []
    ligands7 = []
    odorscenes7 = []
    ligands10 = []
    odorscenes10 = []
    ligands15 = []
    odorscenes15 = []
    ligands20 = []
    odorscenes20 = []
    ligands25 = []
    odorscenes25 = []
    ligands30 = []
    odorscenes30 = []
    ligands35 = []
    odorscenes35 = []
    ligands40 = []
    odorscenes40 = []
    ligands45 = []
    odorscenes45 = []
    ligands50 = []
    odorscenes50 = []
    ligands60 = [] ###
    odorscenes60 = []
    ligands70 = []
    odorscenes70 = []
    ligands80 = []
    odorscenes80 = []
    ligands90 = []
    odorscenes90 = []
    ligands100 = []
    odorscenes100 = []
    ligands120 = []
    odorscenes120 = []
    ligands140 = []
    odorscenes140 = []
    ligands160 = []
    odorscenes160 = []
    ligands200 = []
    odorscenes200 = []
    ligands250 = []
    odorscenes250 = []
    ligands300 = []
    odorscenes300 = []
    ligands350 = []
    odorscenes350 = []
    ligands400 = []
    odorscenes400 = []
    '''
    
    #Holds data for odorscene activations
    st = "Odorscenes,"
    st2 = ""
    st3 = ""
    p = 0
    while p < len(epithelium.getRecs()):
        st += "activ " + str(p) + ","
        st2 += "occ " + str(p) + ","
        st3 += "odoAmt " +str(p) + ","
        p += 1
    text = Text(st + st2 +st3[:-1] + '\n', "exp1")
    
    #If c!=1, also hold info about glom activ
    if c!=1:
        string = "Glom,"
        string2 = ""
        p=0
        while p < len(gl):
            string2 += "activ " + str(p) + ","
            p += 1
        text._st2 = string + string2 +'\n'
    
    
    
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400] #If change here, change xAxis in expFromRnO
    yaxis = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    odorscenesArray = [[]*size for x in range(len(xaxis))]
    #odorscenesArray = [[]]
    pdfOdorLocsInput = []
    affs = np.array([])
    effs = np.array([])
    

    pdfOdorLocsInput2 = []
    affs2 = np.array([])
    effs2 = np.array([])

    dns = []
    #rep = 10.0
    rep = ANGLES_REP


    
    ligandsArray =[]    
    
    #creating ligand  and odorscene lists
    #i = 0
    #while i < size:
    for i in range(size):    
        #k = 0
        #while k < len(xaxis):
            #j = xaxis[k]
        for k, j in enumerate(xaxis):
            n = 0
            while n < j: 
                odor = Ligand(n, createLoc(qspace), conc)
                #ligandsArray.append(Ligand(i, createLoc(qspace), conc))
                ligandsArray.append(odor)
                pdfOdorLocsInput.append(odor.getLoc())
                n+=1
            #odorscenesArray[k].append(Odorscene(i, ligandsArray[k]))
            odorscene = Odorscene(k, ligandsArray)
            odorscenesArray[k].append(odorscene)
            
            #prepare pdf inputs for ordors2
            dns = dPsiBarCalcDns(odorscene, r, rep)
            
            #Second odors
            for oriOdor in odorscene.getOdors():
                for i in range(rep):
                    newLoc = []  #Calculating new location
                    for index, dnItem in enumerate(dns):    
                        newLoc.append(oriOdor.getLoc()[index] + dnItem)
                    newOdor = Ligand(oriOdor.getId(), newLoc, oriOdor.getConc())
                    pdfOdorLocsInput2.append(newOdor.getLoc())
                    oriOdor.appendToOdors2(newOdor)
                    
            
            #text._st += "Odorscene"+str(k+1)
            
            
            #this call moves to after affs and effs are calc'ed and populated to odors below 
            #yaxis[k] += dPsiBarCalcAngles(epithelium, odorscenesArray[k][i], r, fixed, text, c, gl)

            #k+=1
            ligandsArray =[]

        #i += 1



    #draw ellispse for all receptors
    drawEllipseGraph(qspace, epithelium, odorscenesArray, useMockData=False)
    

    #print("locsInput size=:" + str(len(pdfOdorLocsInput)))
    #print("locsInput2 size=:" + str(len(pdfOdorLocsInput2)))
    ##############################################################
    #'''
    for rec in epithelium.getRecs():
        affs_rec = mvn.pdf(pdfOdorLocsInput, rec.getMean(), rec.getCovA())
        
        affs_rec = affs_rec / rec.getScale() #Scales it from 0 to 1
        #Now convert gaussian aff to kda
        #startTime = time.time()
        affs_rec = 10**((affs_rec * (peak_affinity - minimum_affinity)) + minimum_affinity) ##peak_affinity etc. are global variables
        #print("time elapsed odor aff:"+ str((time.time() - startTime)))
        #print("aff is " + str(aff))
        
        rec.setAffs(affs_rec)
        affs = np.append(affs,affs_rec)
        
        
        if not fixed:
            effs_rec = mvn.pdf(pdfOdorLocsInput, rec.getMean(), rec.getCovE())
            effs_rec = np.asarray(effs_rec,dtype=np.float) / rec.getEffScale() #Scales it from 0 to 1


            effs = np.append(effs,effs_rec)
            
        else:
            effs_rec = np.repeat(1.0, affs_rec.size)
            effs = np.repeat(1.0, affs.size)
        rec.setEffs(effs_rec)



        # now do odors2 calc
        affs_rec2 = mvn.pdf(pdfOdorLocsInput2, rec.getMean(), rec.getCovA())
        
        affs_rec2 = affs_rec2 / rec.getScale() #Scales it from 0 to 1
        #Now convert gaussian aff to kda
        #startTime = time.time()
        affs_rec2 = 10**((affs_rec2 * (peak_affinity - minimum_affinity)) + minimum_affinity) ##peak_affinity etc. are global variables
        #print("time elapsed odor aff:"+ str((time.time() - startTime)))
        #print("aff is " + str(aff))
        
        #rec.setAffs(affs_rec2)
        affs2 = np.append(affs2,affs_rec2)
        
        
        if not fixed:
            effs_rec2 = mvn.pdf(pdfOdorLocsInput2, rec.getMean(), rec.getCovE())
            effs_rec2 = np.asarray(effs_rec2,dtype=np.float) / rec.getEffScale() #Scales it from 0 to 1


            effs2 = np.append(effs2,effs_rec2)
            
        else:
            effs_rec2 = np.repeat(1.0, affs_rec2.size)
            effs2 = np.repeat(1.0, affs2.size)


        #rec.setEffs(effs_rec)
           
    #print("affs size=:" + str(affs.size))
    #print("effs size=:" + str(effs.size))

    #print("affs2 size=:" + str(affs2.size))
    #print("effs2 size=:" + str(effs2.size))
            
            #effs = np.asarray(effs,dtype=np.float) / rec.getEffScale() #Scales it from 0 to 1
    #affs = np.asarray(affs,dtype=np.float)
    #effs = np.asarray(effs,dtype=np.float)       


    #zAffs = []

    locXaxis = []
    locYaxis = []

    vi = 0
    vi2 = 0
    for i in range(size):
        for k, j in enumerate(xaxis):
            #saveOdorscene(odorscenesArray[k][i], "Saved Odors" + str(qspace.getSize()[0]) + purp)

            for odor in odorscenesArray[k][i].getOdors(): #odorscenesArray[k][i].getOdors()
                for li, loc in enumerate(odor.getLoc()):
                    if li == 0:
                        locXaxis.append(loc)
                    if li == 1:    
                        locYaxis.append(loc)
                """
                for odor2 in odor.getOdors2():
    
                    for li, loc in enumerate(odor2.getLoc()):
                        if li == 0:
                            locXaxis.append(loc)
                        if li == 1:    
                            locYaxis.append(loc)
                """

            #n = 0
            #while n < j: 
                #print("vi=" + str(vi))
                #print("affs size," + str(affs[vi]))
                #print("effs size," + str(effs[vi]))
                for rec in epithelium.getRecs():
                    
                    odor.appendToAffs(float(affs[vi]))
                    odor.appendToEffs(float(effs[vi]))
                    vi+=1
                    
                    #now set resuts to ordor2
                    for odor2 in odor.getOdors2():
                        odor2.appendToAffs(float(affs2[vi2]))
                        odor2.appendToEffs(float(effs2[vi2]))
                        vi2+=1
                '''
                odor.setAff(float(affs[vi]))
                odor.setEff(float(effs[vi]))
                
                vi+=1
                '''
            text._st += "Odorscene"+str(k+1)    
            yaxis[k] += dPsiBarCalcAngles(epithelium, odorscenesArray[k][i], r, fixed, text, c, gl)


            
    '''
    odor.setAff(float(aff))
    if fixed:
        odor.setEff(1.0)
    else:
        eff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovE())
        eff = float(eff) / effScale #Scales it from 0 to 1
        odor.setEff(eff)
    odors.append(odor)
    return odor.getConc()/odor._aff
    '''

        
    '''
    #Holds data for odorscene activations
    st = "Odorscenes,"
    st2 = ""
    st3 = ""
    p = 0
    while p < len(epithelium.getRecs()):
        st += "activ " + str(p) + ","
        st2 += "occ " + str(p) + ","
        st3 += "odoAmt " +str(p) + ","
        p += 1
    text = Text(st + st2 +st3[:-1] + '\n', "exp1")
    
    #If c!=1, also hold info about glom activ
    if c!=1:
        string = "Glom,"
        string2 = ""
        p=0
        while p < len(gl):
            string2 += "activ " + str(p) + ","
            p += 1
        text._st2 = string + string2 +'\n'
    '''

    '''    
    #Calculate averages of dPsiBar
    i = 0
    while i < size:
        print("bebinning of " + str(i))

        text._st += "Odorscene1"
        yaxis[0] += dPsiBarCalcAngles(epithelium, odorscenes1[i], r, fixed, text, c, gl)
        print("after " + str(yaxis[0]))

        text._st += "Odorscene2"
        yaxis[1] += dPsiBarCalcAngles(epithelium, odorscenes2[i], r, fixed, text, c, gl)
        print("after " + str(yaxis[1]))


        text._st += "Odorscene3"
        yaxis[2] += dPsiBarCalcAngles(epithelium, odorscenes3[i], r, fixed, text, c, gl)
        print("after " + str(yaxis[2]))

        text._st += "Odorscene4"
        yaxis[3] += dPsiBarCalcAngles(epithelium, odorscenes4[i], r, fixed, text, c, gl)
        print("after " + str(yaxis[3]))

        text._st += "Odorscene5"
        yaxis[4] += dPsiBarCalcAngles(epithelium, odorscenes5[i], r, fixed, text, c, gl)
        text._st += "Odorscene7"
        yaxis[5] += dPsiBarCalcAngles(epithelium, odorscenes7[i], r, fixed, text, c, gl)
        print("after " + str(yaxis[5]))
        
        text._st += "Odorscene10"
        d = dPsiBarCalcAngles(epithelium, odorscenes10[i], r, fixed, text, c, gl)
        print("dPsi for odorscene10: " + str(d))
        yaxis[6] += d
        text._st += "Odorscene15"
        yaxis[7] += dPsiBarCalcAngles(epithelium, odorscenes15[i], r, fixed, text, c, gl)
        text._st += "Odorscene20"
        yaxis[8] += dPsiBarCalcAngles(epithelium, odorscenes20[i], r, fixed, text, c, gl)
        text._st += "Odorscene25"
        yaxis[9] += dPsiBarCalcAngles(epithelium, odorscenes25[i], r, fixed, text, c, gl)
        text._st += "Odorscene30"
        yaxis[10] += dPsiBarCalcAngles(epithelium, odorscenes30[i], r, fixed, text, c, gl)
        text._st += "Odorscene35"
        yaxis[11] += dPsiBarCalcAngles(epithelium, odorscenes35[i], r, fixed, text, c, gl)
        text._st += "Odorscene40"
        yaxis[12] += dPsiBarCalcAngles(epithelium, odorscenes40[i], r, fixed, text, c, gl)
        text._st += "Odorscene45"
        yaxis[13] += dPsiBarCalcAngles(epithelium, odorscenes45[i], r, fixed, text, c, gl)
        text._st += "Odorscene50"
        yaxis[14] += dPsiBarCalcAngles(epithelium, odorscenes50[i], r, fixed, text, c, gl)
        text._st += "Odorscene60"
        yaxis[15] += dPsiBarCalcAngles(epithelium, odorscenes60[i], r, fixed, text, c, gl)
        text._st += "Odorscene70"
        yaxis[16] += dPsiBarCalcAngles(epithelium, odorscenes70[i], r, fixed, text, c, gl)
        print("after " + str(yaxis[16]))
        
        text._st += "Odorscene80"
        yaxis[17] += dPsiBarCalcAngles(epithelium, odorscenes80[i], r, fixed, text, c, gl)
        text._st += "Odorscene90"
        yaxis[18] += dPsiBarCalcAngles(epithelium, odorscenes90[i], r, fixed, text, c, gl)
        text._st += "Odorscene100"
        yaxis[19] += dPsiBarCalcAngles(epithelium, odorscenes100[i], r, fixed, text, c, gl)
        text._st += "Odorscene120"
        yaxis[20] += dPsiBarCalcAngles(epithelium, odorscenes120[i], r, fixed, text, c, gl)
        text._st += "Odorscene140"
        yaxis[21] += dPsiBarCalcAngles(epithelium, odorscenes140[i], r, fixed, text, c, gl)
        text._st += "Odorscene160"
        yaxis[22] += dPsiBarCalcAngles(epithelium, odorscenes160[i], r, fixed, text, c, gl)
        text._st += "Odorscene200"
        yaxis[23] += dPsiBarCalcAngles(epithelium, odorscenes200[i], r, fixed, text, c, gl)
        print("after " + str(yaxis[23]))
        
        text._st += "Odorscene250"
        yaxis[24] += dPsiBarCalcAngles(epithelium, odorscenes250[i], r, fixed, text, c, gl)
        text._st += "Odorscene300"
        yaxis[25] += dPsiBarCalcAngles(epithelium, odorscenes300[i], r, fixed, text, c, gl)
        text._st += "Odorscene350"
        yaxis[26] += dPsiBarCalcAngles(epithelium, odorscenes350[i], r, fixed, text, c, gl)
        text._st += "Odorscene400"
        yaxis[27] += dPsiBarCalcAngles(epithelium, odorscenes400[i], r, fixed, text, c, gl)
        print("i is " + str(i))
        i += 1
        '''
    count = 0
    while count < len(yaxis):
        yaxis[count] = yaxis[count]/float(size)
        count += 1
    print(yaxis)
    
    #Saving Activated Epithelium data in excel
    test = open(excelName + ".csv", "w")
    test.write(text._st)
    test.close
    
    if c != 1:
        test = open("Glom_act with c=" + str(c) + " with " + str(qspace.getSize()[0]) + " qspace.csv", "w")
        test.write(text._st2)
        test.close
    
    #Saving dPsi data in excel
    st = "Odorscenes, dPsiBar" + '\n'
    i = 0
    while i < len(xaxis):
        st += str(xaxis[i]) + "," + str(yaxis[i]) + '\n'
        i += 1
    n = "dPsi, qspace=(0, " + str(qspace.getSize()[0][1]) + ")" + purp
    test = open(n + ".csv", "w")
    test.write(st)
    test.close

    if graphIt:
        plt.plot(xaxis,yaxis, label=labelName)
        plt.legend()
        plt.title(plotTitle)
        plt.xlabel("Number of Ligands")
        plt.ylabel("dPsiBar")
    
        #Set y_axis limit
        axes = plt.gca()
        axes.set_ylim([0,0.1]) #*****Change if using >30 recs
    
        #plt.show()
        pp = PdfPages(pdfName + '.pdf')
        pp.savefig()
        pp.close()
        if close == True:
            plt.close()




    #drawContourGraph(qspace)

    #draw ellispse for all receptors
    #drawEllipseGraph(qspace, epithelium, odorscenesArray, False)


    #draw odor locs
    #drawOdorLocations(locXaxis,locYaxis, qspace, False)

    ##########################################
    #'''
    print("time elapsed each qspace:"+ str((time.time() - startTime)))

def createLoc(qspace):
    """Given a qspace, return a list of randomized numbers (len=dim) within the qspace"""
    loc = []
    for tup in qspace.getSize():
        a1 = tup[0]
        a2 = tup[1]
        loc.append(random.uniform(a1,a2))
    return loc

#NOT IN USE
def drawOdorLocations(locXaxis,locYaxis, qspace, close):
    plt.scatter(locXaxis,locYaxis, s=1, label="Odor Location")

    '''
    zAffs = affs_rec
    X,Y = np.meshgrid(affs_rec, affs_rec)
    Z = np.sqrt(X**2 + Y**2)
    
    plt.contourf(X, Y, Z)
    #plt.contourf(X, Y, Z, 24, alpha=0.75, cmap=plt.cm.hot)

    #C=plt.contour(X, Y, Z, 24, colors="black", linewidth=0.5)
    '''

    #plt.legend()
    plt.title("Odor Locations - QSpace "+ str(qspace.getSize()[0]))
    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")
    
    #plt.show()
    pp = PdfPages("Odor Locations " + str(qspace.getSize()[0]) + '.pdf')
    pp.savefig()
    pp.close()
    if close == True: #No more data to add to the graph
        plt.close()


#NOT IN USE
def drawContourGraph(qspace):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X**2 + Y**2)
    plt.figure()
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')

    plt.legend()
    plt.title("Odor Locations - QSpace "+ str(qspace.getSize()[0]))
    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")
    #Set y_axis limit
    #axes = plt.gca()
    #axes.set_ylim([0,1.0]) #*****Change if using >30 recs
    
    #plt.show()
    pp = PdfPages("Contour " + str(qspace.getSize()[0]) + '.pdf')
    pp.savefig()
    pp.close()
    #if close == True: #No more data to add to the graph
        #plt.close()
    #plt.close()


def runReceptorOdorGraphToolStandAlone():
    print('need to convert mock arrays data into obj of qspace, epitheliumm, odorscenesArray so we can call the method below')
    drawEllipseGraph('', '', '', True)

def drawEllipseGraph(qspace, epithelium, odorscenesArray, useMockData=False):
    #print("odorscene 2nd dim len = " + str(len(odorscenesArray[0])))
    assert params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION > 0, "Standard Deviation number must be greater than 0"
    if useMockData == False and params.USE_MOCK_ODORS_EVEN_WHEN_RUNNING_CALCS == False:
        assert params.ODORSCENE_INDEX >= 0 and params.ODORSCENE_INDEX < len(odorscenesArray), "Odorscene index must be within the range of xaxis"
        assert params.ODORSCENE_REP_NUMBER >= 0 and params.ODORSCENE_REP_NUMBER < ODOR_REPETITIONS, "Odorscene rep number must be within the range 0 - number of repetitions"

    ells_sda = []
    ells_sde = []

    """
    print("xy=" + str(rnd.rand(2)*10))
    print("width=" + str(rnd.rand()))
    print("height=" + str(rnd.rand()))
    print("angle=" + str(rnd.rand()*360))
    """
    # ang = rnd.rand()*360            
    # ang1 = rnd.rand()*360
    # ang2 = rnd.rand()*360
    ang = rnd.rand()
    ang1 = rnd.rand()
    ang2 = rnd.rand()

    if params.USE_MOCK_RECEPTORS_EVEN_WHEN_RUNNING_CALCS or useMockData:
        if params.SHOW_SDA_ELLIPSE:
            ells_sda.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN, width=params.MOCK_RECEPTOR_SDA[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDA[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
            ells_sda.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN1, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang1))
            ells_sda.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN2, width=params.MOCK_RECEPTOR_SDA2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDA2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang2))
        
        #if params.MOCK_RECEPTOR_MEAN[0] + params.MOCK_RECEPTOR_SDA[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION > params.MOCK_QSPACE_DIMENSION[1]:
            print("xy=" + str(params.MOCK_RECEPTOR_MEAN[0] + params.MOCK_RECEPTOR_SDA[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION ))
            print("dim=" + str(params.MOCK_QSPACE_DIMENSION[1]))

            # # left side
            # newMean=[(params.MOCK_QSPACE_DIMENSION[1]-params.MOCK_RECEPTOR_MEAN[0])*-1,params.MOCK_RECEPTOR_MEAN[1]]
            # ells_sda.append(Ellipse(xy=newMean, width=params.MOCK_RECEPTOR_SDA[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang))            
            # # right side
            # newMean=[(params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN[0]),params.MOCK_RECEPTOR_MEAN[1]]
            # ells_sda.append(Ellipse(xy=newMean, width=params.MOCK_RECEPTOR_SDA[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang))
            
            # # Mean 1
            # # left side
            # newMean1left=[(params.MOCK_QSPACE_DIMENSION[1]-params.MOCK_RECEPTOR_MEAN1[0])*-1,params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy=newMean1left, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))            
            # # right side
            # newMean1right=[(params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN1[0]),params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy=newMean1right, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))

            # # bottom side
            # newMean1bottom=[params.MOCK_RECEPTOR_MEAN1[0], params.MOCK_RECEPTOR_MEAN1[1]-params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sda.append(Ellipse(xy=newMean1bottom, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))            
            # # top side
            # newMean1top=[params.MOCK_RECEPTOR_MEAN1[0], params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy=newMean1top, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))

            # # bottom left
            # newMean1bottomLeft=[params.MOCK_RECEPTOR_MEAN1[0]-params.MOCK_QSPACE_DIMENSION[1], params.MOCK_RECEPTOR_MEAN1[1]-params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sda.append(Ellipse(xy=newMean1bottomLeft, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))            

            # # bottom right
            # newMean1bottomRight=[params.MOCK_RECEPTOR_MEAN1[0]+params.MOCK_QSPACE_DIMENSION[1], params.MOCK_RECEPTOR_MEAN1[1]-params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sda.append(Ellipse(xy=newMean1bottomRight, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))            
            
            # # top left
            # newMean1topLeft=[params.MOCK_RECEPTOR_MEAN1[0]-params.MOCK_QSPACE_DIMENSION[1], params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy=newMean1topLeft, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))

            # # top right
            # newMean1topRight=[params.MOCK_RECEPTOR_MEAN1[0]+params.MOCK_QSPACE_DIMENSION[1], params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sda.append(Ellipse(xy=newMean1topRight, width=params.MOCK_RECEPTOR_SDA1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))

            # # Mean 2
            # # left side
            # newMean2left=[(params.MOCK_QSPACE_DIMENSION[1]-params.MOCK_RECEPTOR_MEAN2[0])*-1,params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sda.append(Ellipse(xy=newMean2left, width=params.MOCK_RECEPTOR_SDA2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang2))            
            # # right side
            # newMean2right=[(params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN2[0]),params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sda.append(Ellipse(xy=newMean2right, width=params.MOCK_RECEPTOR_SDA2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang2))

            # # bottom side
            # newMean2bottom=[params.MOCK_RECEPTOR_MEAN2[0], params.MOCK_RECEPTOR_MEAN2[1]-params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sda.append(Ellipse(xy=newMean2bottom, width=params.MOCK_RECEPTOR_SDA2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang2))            
            # # top side
            # newMean2top=[params.MOCK_RECEPTOR_MEAN2[0], params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sda.append(Ellipse(xy=newMean2top, width=params.MOCK_RECEPTOR_SDA2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDA2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang2))

        if params.SHOW_SDE_ELLIPSE:
            ells_sde.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN, width=params.MOCK_RECEPTOR_SDE[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDE[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
            ells_sde.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN1, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
            ells_sde.append(Ellipse(xy=params.MOCK_RECEPTOR_MEAN2, width=params.MOCK_RECEPTOR_SDE2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=params.MOCK_RECEPTOR_SDE2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))

        #if params.MOCK_RECEPTOR_MEAN[0] + params.MOCK_RECEPTOR_SDE[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION > params.MOCK_QSPACE_DIMENSION[1]:
            # # left side
            # newMean=[(params.MOCK_QSPACE_DIMENSION[1]-params.MOCK_RECEPTOR_MEAN[0])*-1,params.MOCK_RECEPTOR_MEAN[1]]
            # ells_sde.append(Ellipse(xy=newMean, width=params.MOCK_RECEPTOR_SDE[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang))            
            # # right side
            # newMean=[(params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN[0]),params.MOCK_RECEPTOR_MEAN[1]]
            # ells_sde.append(Ellipse(xy=newMean, width=params.MOCK_RECEPTOR_SDE[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang))

            # # left side
            # newMean1left=[(params.MOCK_QSPACE_DIMENSION[1]-params.MOCK_RECEPTOR_MEAN1[0])*-1,params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy=newMean1left, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))            
            # # right side
            # newMean1right=[(params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN1[0]),params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy=newMean1right, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))

            # # bottom side
            # newMean1bottom=[params.MOCK_RECEPTOR_MEAN1[0], params.MOCK_RECEPTOR_MEAN1[1]-params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sde.append(Ellipse(xy=newMean1bottom, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))            
            # # top side
            # newMean1top=[params.MOCK_RECEPTOR_MEAN1[0], params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy=newMean1top, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))
            
            # # bottom left
            # newMean1bottomLeft=[params.MOCK_RECEPTOR_MEAN1[0]-params.MOCK_QSPACE_DIMENSION[1], params.MOCK_RECEPTOR_MEAN1[1]-params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sde.append(Ellipse(xy=newMean1bottomLeft, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))            

            # # bottom right
            # newMean1bottomRight=[params.MOCK_RECEPTOR_MEAN1[0]+params.MOCK_QSPACE_DIMENSION[1], params.MOCK_RECEPTOR_MEAN1[1]-params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sde.append(Ellipse(xy=newMean1bottomRight, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))            
            
            # # top left
            # newMean1topLeft=[params.MOCK_RECEPTOR_MEAN1[0]-params.MOCK_QSPACE_DIMENSION[1], params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy=newMean1topLeft, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))

            # # top right
            # newMean1topRight=[params.MOCK_RECEPTOR_MEAN1[0]+params.MOCK_QSPACE_DIMENSION[1], params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN1[1]]
            # ells_sde.append(Ellipse(xy=newMean1topRight, width=params.MOCK_RECEPTOR_SDE1[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE1[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang1))

            # # Mean 2
            # # left side
            # newMean2left=[(params.MOCK_QSPACE_DIMENSION[1]-params.MOCK_RECEPTOR_MEAN2[0])*-1,params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sde.append(Ellipse(xy=newMean2left, width=params.MOCK_RECEPTOR_SDE2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang2))            
            # # right side
            # newMean2right=[(params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN2[0]),params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sde.append(Ellipse(xy=newMean2right, width=params.MOCK_RECEPTOR_SDE2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang2))

            # # bottom side
            # newMean2bottom=[params.MOCK_RECEPTOR_MEAN2[0], params.MOCK_RECEPTOR_MEAN2[1]-params.MOCK_QSPACE_DIMENSION[1]]
            # ells_sde.append(Ellipse(xy=newMean2bottom, width=params.MOCK_RECEPTOR_SDE2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang2))            
            # # top side
            # newMean2top=[params.MOCK_RECEPTOR_MEAN2[0], params.MOCK_QSPACE_DIMENSION[1]+params.MOCK_RECEPTOR_MEAN2[1]]
            # ells_sde.append(Ellipse(xy=newMean2top, width=params.MOCK_RECEPTOR_SDE2[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, height=params.MOCK_RECEPTOR_SDE2[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION, angle=ang2))

    else:
        #for rec in epithelium.getRecs():
        ii = -1
        for i, rec in enumerate(epithelium.getRecs()): 
            if params.RECEPTOR_INDEX == 'ALL':
                ii=i
            else:
                ii= params.RECEPTOR_INDEX      
            if i == ii:   
                print("rec" + str(rec.getId()) + ":")
                print("Mean:" + str(rec.getMean()))
                print("SdA:" + str(rec.getSdA()))
                print("Qspace size is " + str(qspace.getSize()[1][1]))
        
                qspaceBoundary = qspace.getSize()[1][1]
                # ang = rnd.rand()*360
                ang = rnd.rand()
                """
                ells.append(Ellipse(xy=rec.getMean(), width=rec.getSdA()[0]*standardDeviationNumber, height=rec.getSdE()[0]*standardDeviationNumber, angle=ang))
                ells.append(Ellipse(xy=rec.getMean(), width=rec.getSdA()[1]*standardDeviationNumber, height=rec.getSdE()[1]*standardDeviationNumber, angle=ang))

                """
                if params.SHOW_SDA_ELLIPSE:
                    ells_sda.append(Ellipse(xy=rec.getMean(), width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    
                    # newMeanLeft = [rec.getMean()[0]-qspaceBoundary, rec.getMean()[1]]
                    # print("newMeanLeft = " + str(newMeanLeft))
                    # print("newMeanLeft x value is " + str(rec.getMean()[0] - qspaceBoundary) + "!!!!! mean is " + str(rec.getMean()) + "!!!!! qspace boundary is " + str(qspaceBoundary))
                    # ells_sda.append(Ellipse(xy=newMeanLeft, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    # newMeanRight = [rec.getMean()[0] + qspaceBoundary, rec.getMean()[1]]
                    # ells_sda.append(Ellipse(xy=newMeanRight, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    # newMeanBottom = [rec.getMean()[0], rec.getMean()[1]-qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMeanBottom, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    # newMeanTop = [rec.getMean()[0], rec.getMean()[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMeanTop, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    # # bottom left
                    # newMean1bottomLeft=[rec.getMean()[0]-qspaceBoundary, rec.getMean()[1]-qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMean1bottomLeft, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))            

                    # # bottom right
                    # newMean1bottomRight=[rec.getMean()[0] + qspaceBoundary, rec.getMean()[1]-qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMean1bottomRight, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))            
            
                    # # top left
                    # newMean1topLeft=[rec.getMean()[0]-qspaceBoundary, rec.getMean()[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMean1topLeft, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))

                    # # top right
                    # newMean1topRight=[rec.getMean()[0] + qspaceBoundary, rec.getMean()[1] + qspaceBoundary]
                    # ells_sda.append(Ellipse(xy=newMean1topRight, width=rec.getSdA()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdA()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))

                    
                if params.SHOW_SDE_ELLIPSE:
                    ells_sde.append(Ellipse(xy=rec.getMean(), width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    
                    # newMeanLeft = [rec.getMean()[0] - qspaceBoundary, rec.getMean()[1]]
                    # ells_sde.append(Ellipse(xy=newMeanLeft, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    # newMeanRight = [rec.getMean()[0] + qspaceBoundary, rec.getMean()[1]]
                    # ells_sde.append(Ellipse(xy=newMeanRight, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    # newMeanBottom = [rec.getMean()[0], rec.getMean()[1] - qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMeanBottom, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    # newMeanTop = [rec.getMean()[0], rec.getMean()[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMeanTop, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))
                    
                    # # bottom left
                    # newMean1bottomLeft=[rec.getMean()[0]-qspaceBoundary, rec.getMean()[1]-qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMean1bottomLeft, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))            

                    # # bottom right
                    # newMean1bottomRight=[rec.getMean()[0] + qspaceBoundary, rec.getMean()[1]-qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMean1bottomRight, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))            
            
                    # # top left
                    # newMean1topLeft=[rec.getMean()[0]-qspaceBoundary, rec.getMean()[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMean1topLeft, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))

                    # # top right
                    # newMean1topRight=[rec.getMean()[0] + qspaceBoundary, rec.getMean()[1] + qspaceBoundary]
                    # ells_sde.append(Ellipse(xy=newMean1topRight, width=rec.getSdE()[0]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, height=rec.getSdE()[1]*params.RECEPTOR_ELLIPSE_STANDARD_DEVIATION*2, angle=ang))


                #ells.append(Ellipse(xy=rnd.rand(2)*10, width=rnd.rand(), height=rnd.rand(), angle=rnd.rand()*360))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    if params.SHOW_SDA_ELLIPSE:
        for e in ells_sda:
            print("The center is " + str(e.center))
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rnd.rand())
            #e.set_facecolor(rnd.rand(3))
            #e.set_facecolor('red')
            e.set_edgecolor(params.SDA_COLOR)
            e.set_fill(params.SDA_FILL)
            if params.SDA_FILL:
                e.set_facecolor(params.SDA_COLOR)
            e.set_label("SDA")
            e.set_linewidth (params.LINE_WIDTH)
            
    
    if params.SHOW_SDE_ELLIPSE:
        for e in ells_sde:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            #e.set_alpha(rnd.rand())
            #e.set_facecolor(rnd.rand(3))
            #e.set_facecolor('blue')
            e.set_edgecolor(params.SDE_COLOR)
            if params.SDE_FILL:
                e.set_facecolor(params.SDE_COLOR)
            e.set_fill(params.SDE_FILL)
            e.set_label("SDE")
            e.set_linewidth (params.LINE_WIDTH)

    if useMockData:
        ax.set_xlim(params.MOCK_QSPACE_DIMENSION)
        ax.set_ylim(params.MOCK_QSPACE_DIMENSION)
        
    else:
        ax.set_xlim(qspace.getSize()[0])
        ax.set_ylim(qspace.getSize()[0])

    """
    sda_patch = patches.Patch(color='red', label='SDA')
    plt.legend(handles=[sda_patch], loc=1)

    sde_patch = patches.Patch(color='blue', label='SDE')
    plt.legend(handles=[sde_patch], loc=4)
    """

    #draw odor locations for specified odorscean
    locXaxis = []
    locYaxis = []
    locSizes = []

    if params.USE_MOCK_ODORS_EVEN_WHEN_RUNNING_CALCS or useMockData:
        locSizes.append((math.log10(params.ODOR_CONCENTRATION)+10)*10)
        #locXaxis = params.MOCK_ODORS_X
        #locYaxis = params.MOCK_ODORS_Y

        for li, loc in enumerate(params.MOCK_ODORS):
                #print('odor conc = ' + str(odor.getConc()))
                #print('odor size = ' + str((math.log10(odor.getConc())+10)*10))

                locXaxis.append(loc[0])
                locYaxis.append(loc[1])

    else:    
        for odor in odorscenesArray[params.ODORSCENE_INDEX][params.ODORSCENE_REP_NUMBER].getOdors(): #odorscenesArray[k][i].getOdors()
            for li, loc in enumerate(odor.getLoc()):
                locSizes.append((math.log10(odor.getConc())+10)*10)
                #print('odor conc = ' + str(odor.getConc()))
                #print('odor size = ' + str((math.log10(odor.getConc())+10)*10))

                if li == 0:
                    locXaxis.append(loc)
                if li == 1:    
                    locYaxis.append(loc)
    #plt.scatter(locXaxis,locYaxis, s=100, c='black')
    plt.scatter(locXaxis,locYaxis, s=locSizes, c=params.ODOR_COLOR)


    #plt.legend()
    #plt.title("Receptors - QSpace "+ str(qspace.getSize()[0])+ "Std Dev "+ str(params.ODORSCENE_REP_NUMBER))
    plt.title(params.GRAPH_TITLE)
    plt.xlabel(params.XLABEL)
    plt.ylabel(params.YLABEL)

    #plt.show()

    if useMockData:
        pp = PdfPages(params.GRAPH_FILE_NAME + str(params.MOCK_QSPACE_DIMENSION) + '.pdf')    
    else:    
        pp = PdfPages(params.GRAPH_FILE_NAME + str(qspace.getSize()[0]) + '.pdf')
        
    pp.savefig()
    pp.close()

    #Not closing it will add odor locations to it
    plt.close()
    
def graphFromExcel(name, xaxis, numRecs, labelName, titleName, pdfName, toggle, rep=10.0, close=False):
    """Given a CSV file from dpsiBarSaturation, create a graph of average receptor
    activation vs num of ligands.
    numRecs=the amount of receptors used in dpsiBarSaturation code(number
    of activation levels etc in excel doc)
    **xaxis=list of numbers that correspond to the odoptes in each odorscene
    Ex: If odorscene with 1 ligand all the way to odorscene with 10 ligands:
        [1,2,3,4,5,6,7,8,9,10]
    name is a string that represent an existing CSV file created from dpsiBarSaturation
    rep = # of repetitions that occur in dpsiBarSaturation
    toggle = "Act" or "Occ"
    close = True when calling graph for last time. Else, False"""
    length = len(xaxis)
    
    text = open(name)
    activ = []
    for x in xaxis:
        activ.append(0)
    
    i= 0
    for line in text:
        if i > 0 and i <= (length*rep): #excludes top line
            #Add all the numbers after comma 1 before comma numRecs+1 and store it
            activ[(i-1)%length] += _parseAndSum(line, numRecs, toggle)
        i += 1
    
    #Iterate through activ and divide everything by float(numRecs)*rep to get avg
    ind = 0
    while ind < length:
        activ[ind] = activ[ind] / (float(numRecs)*rep)
        ind += 1

    text.close()
    
    print(activ)
    
    plt.plot(xaxis,activ, label=labelName)
    plt.legend()
    plt.title(titleName)
    plt.xlabel("Number of Ligands")
    
    yAxisName = "Avg Rec "
    if "Glom_act with c=" in name:
        yAxisName = "Avg Glom "
    
    plt.ylabel(yAxisName + toggle + " Lvl")
    #Set y_axis limit
    axes = plt.gca()
    axes.set_ylim([0,1.0]) #*****Change if using >30 recs
    
    pp = PdfPages(pdfName + '.pdf')
    pp.savefig()
    pp.close()
    if close == True: #No more data to add to the graph
        plt.close()

def dPsiGraphFromExcel(name, qspace, titleName, pdfName, close=False):
    """Given an excel doc with dPsiBar data (generated in simulation)
    this function returns a valid graph
    name = name of excel file *Don't forget .csv at end
    titleName = title of graph
    pdfName = name of PDF file
    close = Done adding data to graph"""
    
    text = open(name)
    xaxis = []
    yaxis = []
    label = "(0," + str(qspace.getSize()[0][1]) + ") qspace"
    if titleName[-3:] == "dim":
        label = str(len(qspace.getSize())) + "D"
    
    i = 0
    for line in text:
        if i > 0:
            comma = line.find(",")
            xaxis.append(int(line[:comma]))
            yaxis.append(float(line[comma+1:]))
        i+=1

    text.close()
    
    plt.plot(xaxis, yaxis, label=label)
    plt.legend()
    plt.title(titleName)
    plt.xlabel("Number of Ligands")
    plt.ylabel("dPsiBar")
    
    #Set y_axis limit
    axes = plt.gca()
    axes.set_ylim([0,0.12]) #*****Change if using >30 recs
    
    pp = PdfPages(pdfName + '.pdf')
    pp.savefig()
    pp.close()
    if close == True: #No more data to add to the graph
        plt.close()

def dPsiOccActGraphFromExcel(nameDpsi, nameAO, xaxis, numRecs, labelName, titleName, pdfName, color="b", rep=200.0, close=False):
    """Given three excel docs (DpsiBar, Act, Occ) generated from the simulation,
    this function returns a dPsiBar vs Act and Occ graph with a given qspace
    preconditions:
    nameDpsi = valid excel doc name that holds dPsi info with .csv extension
    nameAO = valid excel doc name that holds Act and Occ info with .csv extension
    xaxis = list of ints that resemble num of odorscenes (xaxis in normal dPsi vs Odorscenes graph)"""
    assert nameDpsi[-3:] == "csv", "nameDpsi doesn't have .csv extension"
    dPsi = []
    activ = []
    occ = []
    
    ###extract dPsi info
    text = open(nameDpsi)    
    i = 0
    for line in text:
        if i > 0:
            comma = line.find(",")
            dPsi.append(float(line[comma+1:]))
        i+=1
    text.close()
    
    ###extract act and occ info
    length = len(xaxis)
    text = open(nameAO)
    for x in xaxis:
        activ.append(0)
        occ.append(0)
    i= 0
    for line in text:
        if i > 0 and i <= (length*rep): #excludes top line
            #Add all the numbers after comma 1 before comma numRecs+1 and store it
            activ[(i-1)%length] += _parseAndSum(line, numRecs, "Act")
            occ[(i-1)%length] += _parseAndSum(line, numRecs, "Occ")
        i += 1

    #Iterate through and divide everything by float(numRecs)*rep to get avg
    ind = 0
    while ind < length:
        activ[ind] = activ[ind] / (float(numRecs)*rep)
        occ[ind] = occ[ind] / (float(numRecs)*rep)
        ind += 1
    text.close()
    
    #Graph it
    plt.plot(occ, dPsi, '-.', color=color, label=labelName + ", occ")
    plt.plot(activ, dPsi, color=color, label=labelName + ", act")
    plt.legend()
    plt.title(titleName)
    plt.xlabel("Occ and Act")
    plt.ylabel("dPsiBar")
    
    #Set axes limits
    axes = plt.gca()
    axes.set_ylim([0,0.1])
    axes.set_xlim([0,1])
    
    pp = PdfPages(pdfName + '.pdf')
    pp.savefig()
    pp.close()
    if close == True: #No more data to add to the graph
        plt.close()


def _parseAndSum(line, numCommas, toggle):
    """Given a line from an excel csv file, add all the numbers
    after comma1 before numCommas+1.
    Example: if numCommas = 30, then summate all numbers between comma 1 and comma 31
    If activ=True, than 
    Return summation"""
    Sum = 0
    track = 0

    while track < numCommas:
        comma1 = line.find(",")
        comma2 = line.find(",", comma1+1)
        Sum += float(line[comma1+1:comma2])
        
        line = line[comma2:]
        track += 1

    if toggle=="Occ":
        Sum = 0
        track = 0
        while track < numCommas:
            comma1 = line.find(",")
            comma2 = line.find(",", comma1+1)
            Sum += float(line[comma1+1:comma2])
        
            line = line[comma2:]
            track += 1
        
    return Sum


def recDensityDpsiGraph(r, qspace, odorscene, dim, name, labelName, excelName, sd=.5, fixed=False):
    """This function uses a qspace with different # of rec to create different
    receptor densities (receptors are all equally spaced within qspace). Odorscene
    will be adjusted to fit into qspace. Return graph of different dPsi_Bar values
    vs dist between receptors (opp of density).
    r is distance ligands 'move' when calc dPsiBar."""
    recDist = []     #x_axis
    dPsiValues = []  #y_axis
    
    #Create a list of receptors
    receptors = []
    receptors.append(35**2)
    receptors.append(30**2)
    i = 20
    while i > 1:
        receptors.append(i**2)
        i -= 1

    print(receptors)
    text = Text("Receptors, Activ_Lvl, Occ, Num_Odo" + '\n', "exp2")
    
    #Calculate values for graph for each qspace
    amt = 0
    for n in receptors:
        text._st += str(n) + " recs" + '\n'
        
        x = recInQspace(n, dim, qspace, sd) #creating uniformly spread receptor field (epi) based on qspace
        dist = x[1]
        epi = x[0]
            
        dPsibar = dPsiBarCalcAngles(epi, odorscene, r, fixed, text)
        recDist.append(dist)
        dPsiValues.append(dPsibar)
        print(amt)
        amt += 1
    
    #Store data in csv file
    test = open(excelName + ".csv", "w")
    test.write(text._st)
    test.close
    
    #Plot graph ###Figure out how to label graphs!!
    plt.plot(recDist,dPsiValues, label=labelName)
    plt.legend()
    plt.title("receptor distance vs dPsi: varying ligands2")
    plt.xlabel("Distance between receptors")
    plt.ylabel("dPsiBar")
    #plt.show()
    pp = PdfPages(name + '.pdf')
    pp.savefig()
    
def recInQspace(n, dimen, qspace, sd=.5):
    """Given n number of receptors and qspace, returns an epithelium with
    receptors at equally incremented distances from one another and the
    distance between them.
    Precondition: qspace must be a square that has origin at 0"""
    
    dimen = float(dimen)
    n = float(n)
    length = float(qspace.getSize()[0][1])
    chgDim = dimen
    numInRow = math.floor(n**(1/dimen))
    safe = numInRow**dimen
    resetChgDim = (numInRow**(dimen-1)) + 1
    dHigh = length / (math.floor(n**(1/dimen)))
    dLow = length / (math.ceil(n**(1/dimen)))
    switch = safe + 1
    dimen = int(dimen)
    
    coord = []
    i = 0
    while i < n+1:
        j = 0
        coord.append([])
        while j < dimen+1:
            coord[i].append(0)
            j += 1
        i += 1
    
    repeat = True
    while repeat:
        switch2 = switch
        r = 2
        c = 1
        repeat = False
        d = dimen
        while r <= n:
            k = 1
            while k <= dimen:
                coord[r][k] = copy.deepcopy(coord[r-1][k])
                k += 1
            if (r > switch and d>=chgDim):
                coord[r][d] += dLow
            else:
                coord[r][d] += dHigh
            while coord[r][d] >= length:
                coord[r][d] = 0
                d -= 1
                if d == 0:
                    repeat = True
                    switch -= numInRow
                    resetChgDim -= 1
                    if resetChgDim == 0:
                        chgDim -= 1
                        resetChgDim = (numInRow**(chgDim - 1.0)) + 1
                    break
                if (r > switch and d>=chgDim):
                    coord[r][d] += dLow
                else:
                    coord[r][d] += dHigh
                if (coord[r][d] < length):
                    d = dimen
            if repeat:
                break
            r+=1
    n = int(n)
    numLastRow = 0
    last = coord[n][dimen-1]
    while last == coord[n-numLastRow][dimen-1]:
        numLastRow += 1
    newD = length/float(numLastRow)
    count = n - numLastRow + 2
    while count <= n:
        coord[count][dimen] = coord[count-1][dimen] + newD
        count += 1
    
    avgDist = 0
    i = 2
    while i <= n:
        a = dimen
        while coord[i][a] == 0:
            a -= 1
        avgDist += coord[i][a] - coord[i-1][a]
        i += 1
    
    avgDist = float(avgDist) / (float(n)-1)
    
    recs = []
    i = 1
    while i <= n:
        pos = 1
        loc = []
        aff = []
        while pos <= dimen:
            loc.append(coord[i][pos])
            aff.append(sd)
            pos += 1
        recs.append(Receptor(i, loc, aff, aff))
        i += 1
    
    #print(coord)
    #print(avgDist)
    return [Epithelium(recs), avgDist]

def recDensityDpsiGraphRandomized(r, qspace, odorscene, dim, name, fixed=False):
    """Returns graph of dPsi vs # of receptors in a given qspace. Values
    are averaged multiple times to get accurate results. qspace is constant
    while number of receptors varies.
    r is distance ligands 'move' when calc dPsiBar.
    This is different from recDensityDpsiGraph since it doesn't equally allign
    receptors in qspace. Just randomly distributes it and calculated multiple times to get an average."""
    #Create xAxis
    receptorNum = []
    i = 2
    while i <10:
        receptorNum.append(i)
        i += 2
    receptorNum.append(i)
    i += 40
    while i < 100:
        receptorNum.append(i)
        i += 50
    while i <= 1000:
        receptorNum.append(i)
        i+=100

    #Create yAxis template
    dPsi = []
    i=0
    while i < len(receptorNum):
        dPsi.append(0)
        i+=1
    print(receptorNum)

    repeats = 0
    text = Text("Receptors, Activ_Lvl, Occ, Num_Odo" + '\n', "exp2")
    while repeats < 10:
        num=0
        while num < len(receptorNum):
            text._st += "Rec # " + str(receptorNum[num]) + "\n"
            epi = createEpithelium(receptorNum[num], dim, qspace, scale=[.5,1.5])
            dPsi[num] += dPsiBarCalcAngles(epi, odorscene, r, fixed, text)
            num+=1
            print(num)
        print(repeats)
        text._st += "Repeat again" + "\n"
        repeats += 1
    #Average the dPsi calculations
    i = 0
    while i < len(dPsi):
        dPsi[i] = dPsi[i] / 10.0
        i += 1
    
    #Store data in csv file
    test = open(name + ".csv", "w")
    test.write(text._st)
    test.close
    

    plt.plot(receptorNum,dPsi)
    plt.title("dPsi vs # of Receptors")
    plt.xlabel("Number of Receptors")
    plt.ylabel("dPsiBar")
    plt.show()

def glomRecConnNew(recs, gl, c=9, conn=[]):
    """New function that deploys gl into olfactory bulb space (numRow X numCol)
    and connects to primary and secondary recs with given weights.
    c = num of recs connected to each glom
    conn = Used for random assignment to ensure dPsi is calculated using odors with identical connections
    The following variables are global variables:
    glom_penetrance = primary weight
    s_weights = a list of floats for the remaining weights. If empty, the rest of
    the weights are (1-p_weight)/(c-1)
    Preconditons: # of recs in epi == numRow*numCol == len(gl)
                  if constant=True then c must be 9"""
    assert len(gl) == len(recs), "# of recs != # of glom"
    assert len(gl) == numRow*numCol, "Glomeruli don't fit in space. Change numRow and numCol global variables on top of RnO.py"
    assert len(s_weights) == 0 or c-1, "weights is an incorrect length"

    i = 0
    row = 0
    
    #Deploy each glom into space and add primary receptor
    while row < numRow:
        col = 0
        while col < numCol:
            gl[i]._loc = [row,col]
            gl[i]._recConn[recs[i]] = glom_penetrance
            col += 1
            i += 1
        row += 1
    
    if s_weights == []:
        w = (float(1-glom_penetrance)) / (float(c-1))
        for i in range(c):
            s_weights.append(w)

    ####Loop through glom and attach remaining recs
    conn = attachSecondaryRecs(gl, recs, c, conn)


    #Activate each glom given rec connections
    i = 0
    for glom in gl:
        activ = 0.0
        for rec in glom._recConn.keys():
            activ += float(rec._activ )* float(glom._recConn[rec]) #rec Activ * weight of conn
            #print("rec id: " + str(float(rec._id )) + " weight is: " + str(float(glom._recConn[rec])))
        glom.setActiv(min(activ, 1.0))
        #print("ACTIV IS:" + str(min(activ, 1.0)))
    
    return conn

def attachSecondaryRecs(gl, recs, c, conn):
    """Given gl with primary rec attachments, this function attaches the remaining recs
    if constant=true, then attaches surrounding 8 recs, otherwise it's random assignment.
    If conn = [], randomly assign. If conn isn't empty, than use the given conn info to attach recs
    Returns conn"""

    if constant_attachments: #Loop through glom and find all surrounding gloms to add their recs as secondary
        for glom in gl: 
            locations = getLocations(glom.getLoc())
            i = 0
            for glom2 in gl:
                if glom2.getLoc() in locations:
                    for rec in glom2._recConn.keys():
                        if glom2._recConn[rec] == glom_penetrance:
                            glom._recConn[rec] = s_weights[i]
                    i += 1

    else: #Randomly assign secondary rec attachments
        ints = range(0,len(recs))
        
        if conn == []: #No prior connection restrictions
            for glom in gl:
                connections = []
                i = 0
                while i < c-1:
                    index = random.choice(ints)
                    rec = recs[index]
                    while rec in glom._recConn.keys(): #Choose random rec until find one that isn't connected yet
                        index = random.choice(ints)
                        rec = recs[index]
                    glom._recConn[rec] = s_weights[i]
                    connections.append(index)
                    i += 1
                conn.append(connections)

        else: #need glom-rec conn to be identical to past conn
            x = 0
            for glom in gl:
                i = 0
                while i < c-1:
                    glom._recConn[recs[conn[x][i]]] = s_weights[i]
                    i += 1
                x += 1

    return conn


def getLocations(location):
    """Returns a list of 8 locations (2d list) that surround loc (modular
    numRow and numCol to create tourus."""
    x = location[0]
    y = location[1]
    locations = [[x-1,y-1],[x,y-1],[x+1,y-1],[x-1,y],[x+1,y],[x-1,y+1],[x,y+1],[x+1,y+1]]
    for loc in locations:
        loc[0] = loc[0]%numRow
        loc[1] = loc[1]%numCol
    return locations
    
    
##############################To make everything more efficient - at the moment sum of squares calculates the same activ_1
#over and over again when called by dPsi angles (it only calculates a new activ_2) - can make a lot more efficient