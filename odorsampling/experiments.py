# Running experiments on functions in RnO
# Mitchell Gronowitz
# 2015-2017

# Edited by Christopher De Jesus
# Summer 2023

from __future__ import annotations

import time
import multiprocessing
import logging
import math
from dataclasses import dataclass, field
from collections import ChainMap
import inspect

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import multivariate_normal as mvn

from odorsampling import config, layers, utils
from odorsampling.RnO import (
    QSpace, Epithelium, Ligand, Receptor, Odorscene,
    dPsiBarSaturation, dPsiGraphFromExcel, graphFromExcel, dPsiOccActGraphFromExcel, activateGL_QSpace
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Mapping, Sequence, Iterable, Any

logger = logging.Logger(__name__)
utils.default_log_setup(logger)


# TODO: Factor this out
def purpFunction(purpose, aff_sd: tuple[float,float]=(0.5, 1.5), eff_sd: tuple[float, float]=(0.05, 1.0), numRecs=30, c=1, dim=2):
    """Returns a string that can be used for titles etc."""
    
    if purpose == "aff": 
        return ", aff_sd=" + str(aff_sd)
    elif purpose == "eff":
        if len(eff_sd) == 1:
            return ", eff_sd=" + str(eff_sd[0])
        else:
            return ", eff_sd=" + str(eff_sd)
    elif purpose == "c":
        return ", glom_pen=" + str(config.GLOM_PENETRANCE)
    elif purpose == "recs":
        return ", numRecs=" + str(numRecs)
    elif purpose == "redAff":
        num = str(8 + config.PEAK_AFFINITY)
        return ", redAff by 10^" + num
    elif purpose == "dim":
        return ", dim=" + str(dim)
    else: #purpose == "standard"
        return ""

def makeSimilar(numRecs, aff_sd: tuple[float, float], eff_sd: tuple[float, float], purpose="eff", qspaces=[4,10,30], dim=2):
    """Creates and saves three epithelium determined by qspaces.
    It keeps aff and eff SD identical and only changes means."""
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, 1, dim)
    
    space = [(0, qspaces[0]) for _ in range(dim)]
    qspace = QSpace(space)
    epith = Epithelium.create(numRecs, dim, qspace, aff_sd, eff_sd) #amt, dim **amt = len(gl) and dim = dim of odorscene
    epith.save("1. SavedEpi_" + str(qspace.size[0]) + purp)
    
    for qspace_ in qspaces[1:]:
        
        space=[(0,qspace_) for _ in range(dim)]
        
        qspace = QSpace(space)
        epith2 = Epithelium.create(numRecs, dim, qspace, aff_sd, eff_sd)
    
        for k, rec in enumerate(epith2.recs):
            rec.sdA = epith.recs[k].sdA
            rec.sdE = epith.recs[k].sdE        
    
        epith2.save("1. SavedEpi_" + str(qspace.size[0]) + purp)
        

@utils.verbose_if_debug
def allGraphsFromExcel(aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4,10,30], purpose="standard", rep=200.0):
    """Given excel docs in correct directories, this creates a dpsiSaturation graph and act and occ graphs"""
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, c, dim)
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400] #xaxis for dPsi vs num of ligands
    
    ####Creating dPsiSaturation graphs
    i = 0
    pdfName = "LigandSat with varying qspaces" + purp
    titleName = "Saturation of dPsiBar" + purp
    qspaceList: list[QSpace] = []
    dPsiName=[]
    end = False
    while i < len(qspaces):
        
        space = []
        j = 0
        while j < dim:
            space.append((0,qspaces[i]))
            j+=1
        qspaceList.append(QSpace(space))
        dPsiName.append("dPsi, qspace=" + str(qspaceList[i].size[0]) + purp + ".csv")

        if i == (len(qspaces)-1):
            end = True
            
        dPsiGraphFromExcel(dPsiName[i], qspaceList[i], titleName, pdfName, end)        
        i += 1

    ####Creating Occ and Rec Act graphs    
     
    for i in range(2):
        if i == 0:
            toggle = "Act"
        else:
            toggle = "Occ"
    
        pdfName = "Rec" + toggle + " vs num of Ligands" + purp
        titleName = "Rec " + toggle + " vs num of Ligands" + purp
        
        k = 0
        labelNames=[]
        excelName=[]
        end = False
        while k < len(qspaces):
            labelNames.append(str(qspaceList[k].size[0]) + " qspace")
            excelName.append("LigandSat with " + str(qspaceList[k].size[0]) + " qspace" + purp)
            
            if k == (len(qspaces)-1):
                end = True

            graphFromExcel(excelName[k] + ".csv", xaxis, numRecs, labelNames[k], titleName, pdfName, toggle, rep, end)
            k += 1
            
    ###Extra dPsi vs occ and act graph
    k = 0
    colors = ['b','g','r','c','m']
    end = False
    while k < len(qspaces):
        titleName = "DpsiBar vs Occ and Act" + purp
        pdfName = "DpsiBar vs Occ and Act" + purp
        if k == (len(qspaces)-1):
            end = True
        dPsiOccActGraphFromExcel(dPsiName[k], excelName[k]+".csv", xaxis, numRecs, labelNames[k], titleName, pdfName, colors[k%5], rep, end)
        k += 1
            

    
    if c!=1:
        pdfName = "Glom Act vs num of Ligands" + purp
        titleName = "Glom Act vs num of Ligands" + purp
        
        k = 0
        end = False
        while k < len(qspaces):
            if k == (len(qspaces)-1):
                end = True
            name = "Glom_act with c=" + str(c) + " with " + str(qspaceList[k].size[0]) + " qspace"
            graphFromExcel(name + ".csv", xaxis, numRecs, labelNames[k], titleName, pdfName, "Act", rep, end)
            k += 1


def psi_bar_saturation_dim(dims, fixed=False, aff_sd=[.5,1.5], eff_sd=[.05,1.0], numRecs=30, c=1, graph=False):
    """Runs 4 simulations of differing dimensions determined by dims all with (0,4) qspace.
    Since each simulation has an added dimension, it wasn't possible
    to make the epithelium identical. Therefore, all means, aff and eff
    are randomized between a constant distribution.
    Returns a dPsiBar graph with dimensions specified in dims. Also returns act and occ graphs
    and excel docs with details.
    
    dims= list of ints that represent dimension.
    fixed = True if want eff=1
    
    Can uncomment loadEpithelium lines if you have saved epi excel docs"""
    
    pdfName = "LigandSat with varying dimensions"
    plotTitle = "Saturation of dPsiBar, varying dim"
    labels = []
    excels = []
    end = False
    index = 0
    for dim in dims:

        space = []
        i = 0
        while i < dim:
            space.append((0,4))
            i+=1
        qspace = QSpace(space)
        #epith = loadEpithelium("SavedEpi_(0,4)_" + str(dim) + "Dim.csv")
        epith = Epithelium.create(numRecs, dim, qspace, aff_sd, eff_sd)
        Epithelium.save(epith, "1. SavedEpi_(0, 4), dim=" + str(dim))
        
        labels.append(str(dim) + "D")
        excels.append("LigandSat with (0, 4) qspace, dim=" + str(dim))
        
        if index == (len(dims) - 1):
            end = True

        dPsiBarSaturation(epith, .01, qspace, pdfName, labels[index], excels[index], fixed, c, plotTitle, end,  ', dim=' + str(dim), False)
        index += 1
    
    if graph:
        dimAllGraphsFromExcel(numRecs, dims)
    

        

def dimAllGraphsFromExcel(numRecs=30, dims=[2,3,4,5], rep=200.0):
    """Same as function above, but adjusted slightly to account for different dimensions."""
    
    ####Creating dPsiSaturation graphs
    i = 0
    pdfName = "LigandSat with varying dim"
    titleName = "Saturation of dPsiBar, varying dim"
    dPsiName = []
    space=[]
    end = False
    for dim in dims:
        space = []
        k = 0
        while k < dim:
            space.append((0,4))
            k+=1
        qspace = QSpace(space)

        dPsiName.append("dPsi, qspace=" + str(qspace.size[0]) + ", dim=" + str(dim) + ".csv")

        if i == (len(dims)-1):
            end = True
            
        dPsiGraphFromExcel(dPsiName[i], qspace, titleName, pdfName, end)

        i += 1

    ####Creating Occ and Rec Act graphs    
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400]
    excelName=[]
    labelNames=[]
    
    for i in range(2):
        if i == 0:
            toggle = "Act"
        else:
            toggle = "Occ"
    
        pdfName = "Rec" + toggle + " vs num of Ligands, varying dim"
        titleName = "Rec " + toggle + " vs num of Ligands, varying dim"
        
        k = 0
        end = False
        while k < len(dims):
            
            if k == (len(dims)-1):
                end = True

            labelNames.append(str(dims[k]) + "D")
            excelName.append("LigandSat with (0, 4) qspace, dim=" + str(dims[k]))

            graphFromExcel(excelName[k] + ".csv", xaxis, numRecs, labelNames[k], titleName, pdfName, toggle, rep, end)
            k += 1
    
    ###Extra dPsi vs occ and act graph
    k = 0
    colors = ['b','g','r','c','m']
    end = False
    while k < len(dims):
        titleName = "DpsiBar vs Occ and Act" + ", dim=" + str(dim)
        pdfName = "DpsiBar vs Occ and Act" + ", dim=" + str(dim)
        if k == (len(dims)-1):
            end = True
        dPsiOccActGraphFromExcel(dPsiName[k], excelName[k]+".csv", xaxis, numRecs, labelNames[k], titleName, pdfName, colors[k%5], rep, end)
        k += 1


@utils.verbose_if_debug
def psi_bar_saturation(fixed, aff_sd: tuple[float, float] = (0.5, 1.5), eff_sd: tuple[float, float] = (0.05, 1.0), numRecs = 30,
                       c = 1, dim = 2, qspaces=[4,10,30], purpose="standard", graph=False):
    """Runs multiple graphs of given qspaces at one time
    Optional - run makeSimilar, to create epitheliums with equal eff and aff SD's (only rec means differ)
    Otherwise - make sure there are three saved epithelium files with correct names

    Returns a dPsiBar graph with varying qspaces, an act and occ graph, and excel docs with details
    
    fixed = True if want eff = 1
    c = convergence ratio of recs to glom
    purpose = reason for running simulation = either "eff", "aff", "c", "recs", "redAff", "dim" or 'standard'"""
    
    #Run this function if don't already have saved epithelium files to use
    makeSimilar(numRecs, aff_sd, eff_sd, purpose, qspaces, dim)
    
    startTime = time.time()
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, c, dim)
    
    pdfName = "LigandSat with varying qspaces" + purp
    plotTitle = "Saturation of dPsiBar" + purp
    
    i = 0
    labelNames = []
    excelNames = []
    end = False
    while i < len(qspaces):
        
        space = []
        j = 0
        while j < dim:
            space.append((0,qspaces[i]))
            j+=1
        qspace = QSpace(space)
        epith = Epithelium.load("1. SavedEpi_" + str(qspace.size[0]) + purp + ".csv")

        labelNames.append(str(qspace.size[0]) + " qspace")
        excelNames.append("LigandSat with " + str(qspace.size[0]) + " qspace" + purp)
        
        if i == (len(qspaces) - 1):
            end = True

        #epi, dn, qspace, pdfName, labelName, excelName, fixed eff
        dPsiBarSaturation(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, True)
        
        i += 1
        pass

    #Creating Occ and Rec Act graphs
    ###################amt of rep in dPsiSaturation function and xAxis. MUST change if change in function
    rep = config.ODOR_REPETITIONS
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400]
    numRecs = len(epith.recs)
    
    for i in range(2):
        if i == 0:
            toggle = "Act"
        else:
            toggle = "Occ"
    
        pdfName = "Rec" + toggle + " vs num of Ligands" + purp
        titleName = "Rec " + toggle + " vs num of Ligands" + purp
        
        k = 0
        end = False
        while k < len(qspaces):
            if k == (len(qspaces)-1):
                end = True
            graphFromExcel(excelNames[k] + ".csv", xaxis, numRecs, labelNames[k], titleName, pdfName, toggle, rep, end)
            k += 1
    
    if c!=1:
        pdfName = "Glom Act vs num of Ligands" + purp
        titleName = "Glom Act vs num of Ligands" + purp

        k = 0
        end = False
        while k < len(qspaces):
            if k == (len(qspaces)-1):
                end = True
            name = "Glom_act with c=" + str(c) + " with " + str(qspace.size[0]) + " qspace"
            graphFromExcel(name + ".csv", xaxis, numRecs, labelNames[k], titleName, pdfName, "Act", rep, end)
            k += 1

def occ_vs_loc(affList=None):
    """Takes a Receptor instance in a qspace=(0,4) and outputs
    Occupancy vs Location plot for rec with affSD=affList
    Highest affSD is a solid line on the graph"""
    if affList is None:
        affList = [2,1.5,1,.5]
    dim = 1
    qspace = QSpace([(0,4)])
    # FIXME: THESE ARE NOT ODORSCENES EADJKFJASDJ
    odorscenes: list[Ligand] = []  #Create 1600 odorscenes (with 1 ligand each) that span qspace from 0 to 3.9
    gl = layers.GlomLayer.create(1)
    i = 0.0
    ID = 0
    while i < qspace.size[0][1]-.01:
        odo = Ligand(ID, [i], 1e-5) #ID, Loc, Conc
        odorscenes.append(odo)
        ID += 1
        i += .01

    recs: list[Receptor] = []
    for aff in affList:
        recs.append(Receptor(1,[2],[aff],[1])) #Id, mean, sda, sde
 
    index = 0
    for rec in recs:
        df = 0
        location=[]
        occupancy=[]
        labelName = "AffSD=" + str(rec.sdA)
        if max(affList) in rec.sdA:
            line = "-"
        else:
            line = "--"
        for odor in odorscenes:
            aff = mvn.pdf(odor.loc, rec.mean, rec.covA)
            aff = aff / rec.scale #Scales it from 0 to 1
                
            #Now convert gaussian aff to kda
            aff = 10**((aff * (config.PEAK_AFFINITY - config.MIN_AFFINITY)) + config.MIN_AFFINITY) ##peak_affinity etc. are global variables
            
            odor.aff = float(aff)
            df = odor.conc/odor._aff
    
            location.append(odor.loc)
            occ = ( (1) / (1 + ( (odor._aff/odor.conc) * (1 + df - (odor.conc / odor._aff ) ) ) **config.HILL_COEFF) ) # m aka HILL_COEFF = 1
            occupancy.append(occ)

        plt.plot(location,occupancy, line, label=labelName)
        plt.title("Occ vs Loc")
        plt.xlabel("Location")
        plt.ylabel("Occupancy")
        plt.legend()
        pp = PdfPages("OccVsLoc" + '.pdf')
        pp.savefig()
        pp.close()
        
        index+=1
    #plt.show()
    plt.close()
    
def eff_vs_loc(effList=None):
    """Takes a Receptor instance in a full qspace and outputs
    Efficacy vs Location plot for rec with effSD=effList.
    The highest effSD gives a solid line"""
    
    if effList is None:
        effList = [.1,.5,1,2,3]

    dim = 1
    qspace = QSpace([(0,4)])
    odorscenes: list[Ligand] = []  #Create 1600 odorscenes (with 1 ligand each) that span qspace from 0 to 3.9
    gl = layers.GlomLayer.create(1)
    i = 0.0
    ID = 0
    while i < qspace.size[0][1]-.01:
        odo = Ligand(ID, [i], 1e-5) #ID, Loc, Conc
        odorscenes.append(odo)
        ID += 1
        i += .01

    recs: list[Receptor] = []
    for eff in effList:
        recs.append(Receptor(1,[2],[1],[eff])) #Id, mean, sda, sde
        
    index = 0
    for rec in recs:
        # df = 0
        location=[]
        efficacy=[]
        labelName = "EffSD=" + str(rec.sdE)
        effScale = float(mvn.pdf(rec.mean, rec.mean, rec.covE)  )

        if max(effList) in rec.sdE:
            line = "-"
        else:
            line = "--"
        for odor in odorscenes:
            eff = mvn.pdf(odor.loc, rec.mean, rec.covE)
            eff = float(eff) / effScale #Scales it from 0 to 1
    
            location.append(odor.loc)
            efficacy.append(eff)

        plt.plot(location,efficacy, line, label=labelName)
        plt.title("Eff vs Loc")
        plt.xlabel("Location")
        plt.ylabel("EFficacy")
        plt.legend()
        pp = PdfPages("EffVsLoc" + '.pdf')
        pp.savefig()
        pp.close()
        
        index+=1
    #plt.show()
    plt.close()



#####Histogram to show that model works when varying eff and aff
def eff_analysis(effSD, affSD=[2,2], qspace=(0,4), fixed=False):
    """Goal: Show that our model works - varying eff and aff creates agonists etc.    
    Returns one graph with a histogram of number of locations (there is a ligand at each location)
    that activate a receptor to a specific activation and a line graph
    of avg efficacy in each "activation section"
    **qspace argument is converted to actual QSpace in the function. Just input (x,y)
    Preconditions: effSD is the distribution scale for the SD [x1,x2]"""
    
    #Consants
    dim = 2
    qspace = QSpace([qspace, qspace])
    odorscenes: list[Odorscene] = []  #Create 1600 odorscenes (with 1 ligand each) that span qspace from 0,0 to 3.9,3.9
    gl = layers.GlomLayer.create(1)
    i = 0.0
    ID = 0
    while i < qspace.size[0][1]:
        j = 0.0
        while j < qspace.size[1][1]:
            odo = Ligand(ID, [i,j], 1e-5) #ID, Loc, Conc
            odorscenes.append(Odorscene(0,[odo]))
            ID += 1
            j += .1
        i += .1
    
    epi = Epithelium.create(1, dim, qspace, affSD, effSD, True) #Creates an epithelium with 1 rec (and not constant mean)
    logger.debug("Aff sd distr: " + str(epi.recs[0]._sdA))
    logger.debug("eff sd distr: " + str(epi.recs[0]._sdE))
    logger.debug("mean is " +str(epi.recs[0]._mean))
    
    bins = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]
    xAxis2 = [.05,.15,.25,.35,.45,.55,.65,.75,.85,.95]
    yAxis_act = [0]*10
    yAxis_eff = [0]*10

    #Loop through ligands, activate
    #Within loop:
    for odors in odorscenes:
        activateGL_QSpace(epi, odors, gl, fixed) #if fixed=True, eff is fixed at 1
        activ= epi.recs[0].activ 
        index = int(math.floor(activ*10.0))
        
        yAxis_act[index] += 1 #Add 1 to the correct location based on activation
        yAxis_eff[index] += odors.odors[0]._eff

    i = 0
    for elem in yAxis_eff: #divide to get the avg efficacy
        #yAxis_eff[i] = elem/(float(len(odorscenes)))
        yAxis_eff[i] = elem/(float(yAxis_act[i]))
        i +=1

    logger.debug("activation bin: " + str(yAxis_act))
    logger.debug("mean efficacy: " + str(yAxis_eff))
    
    #Hist of activation levels
  
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.bar(bins, yAxis_act, width=.1, color='.7', label="# in bin")
    ax2.plot(xAxis2, yAxis_eff, '-o', color='k', label="Avg eff")
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper center')
    ax1.set_xlabel("Activation bins")
    ax1.set_ylabel('# in bins')
    ax2.set_ylabel("Mean efficacy values", color='k')
    
    if fixed:
        title = "Act with fixed eff=1"
        name = "Act Hist, fixed eff=1"
    else:
        title = "Act with eff sd: " + str(effSD)
        name = "Act Hist, eff=" + str(effSD)
        
    plt.title(title)

    ax1.set_ylim([0,1300])
    ax2.set_ylim([0,1])

    pp = PdfPages(name + '.pdf')
    pp.savefig()
    plt.close()
    pp.close()

def graph_all():
    allGraphsFromExcel()

@dataclass
class Experiment:
    name: str
    funcs: Sequence[Callable]
    arg_maps: Sequence[Sequence[Any]] = field(default_factory=list)
    kwarg_maps: Sequence[Mapping[str, Any]] = field(default_factory=list)
    msg: str = "Performing experiment `%s`..."
    description: str = "No description provided."
    default_args: Sequence[Sequence[Any]] = field(default_factory=list, repr=False, compare=False, kw_only=True)
    """
    Deleted in `__post_init__`.
    """
    default_kwargs: Sequence[Mapping[str, Any]] = field(default_factory=list, repr=False, compare=False, kw_only=True)
    """
    Deleted in `__post_init__`.
    """


    def __post_init__(self):
        # Annoying creation of `dict` instances, change if workaround found. Allows for `arg` shadowing.
        self.arg_maps = [list(ChainMap(dict(enumerate(arg_map)), dict(enumerate(self.default_args[i]))).values()) for i, arg_map in enumerate(self.arg_maps)]
        self.kwarg_maps = [ChainMap(kwarg_map, self.default_kwargs[i]) for i, kwarg_map in enumerate(self.kwarg_maps)]
        del self.default_args, self.default_kwargs

    def __call__(self):
        print(self.msg % (self.name))
        results = []
        for i, func in enumerate(self.funcs):
            print(f"  Running function `{func.__name__}`...")
            results.append(func(*self.arg_maps[i], **self.kwarg_maps[i]))
        return results

class NonExperiment(Experiment):
    """
    For non-experiment procedures between tests.
    """
    msg: str = "Performing cleanup operation #%s..."

def validate_func_map(mapping: Mapping[str, Mapping[str, list|dict[str,Any]]]) -> dict[str, dict[str, list|dict[str, Any]]]:
    """
    Raises a ValueError if the mapping contains invalid function names/data.
    """
    for name, func_info in mapping.items():
        try:
            func = globals()[name]
            func_sig = inspect.signature(func)
        except KeyError as e:
            raise ValueError(f"Function '{name}' not found in globals().") from e
        except TypeError as e:
            raise ValueError(f"'Function' '{name}' is not callable.") from e
        try:
            arg_co = len(func_info['default_args'])
            kwarg_names = list(func_info['default_kwargs'].keys())
        except KeyError as e:
            raise ValueError(f"Missing key '{e.args[0]}' for function '{name}'.") from e
        kwarg_co = len(kwarg_names)

        if arg_co+kwarg_co > len(func_sig.parameters):
            raise ValueError(f"Too many arguments for function '{name}'.", f"{arg_co+kwarg_co} > {len(func_sig.parameters)}")
        if not all(name in func_sig.parameters for name in kwarg_names):
            raise ValueError(f"Invalid keyword argument(s) for function '{name}'.")
    # Ensure dict, in case of other Mapping type
    return {k:dict(v) for k,v in mapping.items()}
        
def validate_exp_map(mappings: Iterable[Mapping[str, Any]], default_values: Mapping[str, dict]) -> dict[str, Experiment]:
    """
    Raises a ValueError if the mapping contains invalid experiment names/data.
    """
    # for reference:
    # mappings
    # [
    #   {
    #       'id': 'name',
    #       'name': 'name',
    #       'functions': ['func_name'],
    #       'args': [
    #           [1,2,3],
    #       ],
    #       'kwargs': [
    #           {}
    #       ],
    #       'description': 'description',
    #   }
    # ]
    # default_values
    # {
        # 'func_name': {
        #   'default_args': [],
        #   'default_kwargs': {},
        # }
    # }

    try:
        return {
            exp['id']: Experiment(
                exp['name'], [globals()[func] for func in exp['functions']], exp['args'], exp['kwargs'], description=exp['description'],
                default_args=[default_values[func]['default_args'] for func in exp['functions']],
                default_kwargs=[default_values[func]['default_kwargs'] for func in exp['functions']]
            ) for exp in mappings
        }
    except KeyError as e:
        raise ValueError(f"Missing key '{e.args[0]}' for experiment mapping: {mappings}") from e
    



def test(to_test: Iterable[Experiment]):
    print(f"Random seed is {utils.RNG.bit_generator.seed_seq}")
    return [
        experiment() for experiment in to_test
    ]


# No longer works since requires Experiments to be generate from a map.
# if __name__ == "__main__":
#     test()
