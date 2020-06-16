#Running experiments on functions in RnO
#Mitchell Gronowitz
#2015-2017

from RnO import *
import random
import layers
import copy

#####Below are two simulations for dPsiBarSaturation Graphs.
#The first tests different qspaces
#The second tests different dimensions
#makeSimilar is a helper function


def testdPsiBarSaturation_Qspaces(fixed, aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4,10,30], purpose="standard"):
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
        epith = loadEpithelium("1. SavedEpi_" + str(qspace.getSize()[0]) + purp + ".csv")

        labelNames.append(str(qspace.getSize()[0]) + " qspace")
        excelNames.append("LigandSat with " + str(qspace.getSize()[0]) + " qspace" + purp)
        
        if i == (len(qspaces) - 1):
            end = True

        #epi, dn, qspace, pdfName, labelName, excelName, fixed eff
        dPsiBarSaturation(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, True)
        
        i += 1
        print "Graph #" + str(i) + ": " + str((time.time() - startTime) / 60.0 ) + " minutes"

    #Creating Occ and Rec Act graphs
    ###################amt of rep in dPsiSaturation function and xAxis. MUST change if change in function
    rep = ODOR_REPETITIONS        
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400]
    numRecs = len(epith.getRecs())
    
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
            name = "Glom_act with c=" + str(c) + " with " + str(qspace.getSize()[0]) + " qspace"
            graphFromExcel(name + ".csv", xaxis, numRecs, labelNames[k], titleName, pdfName, "Act", rep, end)
            k += 1
    
    print "Overall time: " + str((time.time() - startTime) / 60.0 ) + " minutes"


def testdPsiBarSaturationDim(dims, fixed=False, aff_sd=[.5,1.5], eff_sd=[.05,1.0], numRecs=30, c=1):
    """Runs simulations of differing dimensions determined by dims all with (0,4) qspace.
    Since each simulation has an added dimension, it wasn't possible
    to make the epithelium identical. Therefore, all means, aff and eff
    are randomized between a constant distribution.
    Returns a dPsiBar graph with all dimensions in dims shown. Also returns act and occ graphs
    and excel docs with details.
    
    dims= list of ints that represent dimension.
    fixed = True if want eff=1
    
    Can uncomment loadEpithelium lines if you have saved epi excel docs"""
    
    startTime = time.time()
    
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
        epith = createEpithelium(numRecs, dim, qspace, aff_sd, eff_sd)
        saveEpithelium(epith, "1. SavedEpi_(0,4), dim=" + str(dim))
        
        labels.append(str(dim) + "D")
        excels.append("LigandSat with (0, 4) qspace, dim=" + str(dim))
        
        if index == (len(dims) - 1):
            end = True

        dPsiBarSaturation(epith, .01, qspace, pdfName, labels[index], excels[index], fixed, c, plotTitle, end, 'dim=' + str(dim), True)
        index += 1

    
    #Creating Occ and Rec Act graphs
    ###################amt of rep in dPsiSaturation function and xAxis. MUST change if change in function
    rep = ODOR_REPETITIONS      
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400]
    
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
                
            graphFromExcel(excels[k] + ".csv", xaxis, numRecs, labels[k], titleName, pdfName, toggle, rep, end)
            k+=1
        
    print "Overall time: " + str((time.time() - startTime) / 60.0 ) + " minutes"

#####Below are functions to create similar epithelium and save them. These
#epithelium can then be used to run a function above.
#1. makeSimilar = to create three new epithelium with standard qspaces (fixed aff and eff SD)
#2. changeOne = given a saved epithelium in (0,4) qspace, creates three new epi
#with standard qspaces and identical means and aff/eff and only changes aff/eff
#3. changeMean = given a saved epithelium, creates an identical epithelium with diff qspace/mean

def makeSimilar(numRecs, aff_sd, eff_sd, purpose="eff", qspaces=[4,10,30], dim=2):
    """Creates and saves three epithelium determined by qspaces.
    It keeps aff and eff SD identical and only changes means."""
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, 1, dim)
    
    space = []
    i = 0
    while i < dim:
        space.append((0,qspaces[0]))
        i+=1
    qspace = QSpace(space)
    epith = createEpithelium(numRecs, dim, qspace, aff_sd, eff_sd) #amt, dim **amt = len(gl) and dim = dim of odorscene
    saveEpithelium(epith, "1. SavedEpi_" + str(qspace.getSize()[0]) + purp)
    
    i = 1
    while i < len(qspaces):
        
        space=[]
        k=0
        while k < dim:
            space.append((0,qspaces[i]))
            k+=1
        qspace = QSpace(space)
        epith2 = createEpithelium(numRecs, dim, qspace, aff_sd, eff_sd)
    
        k = 0
        for rec in epith2.getRecs():
            rec.setSdA(epith.getRecs()[k].getSdA())
            rec.setSdE(epith.getRecs()[k].getSdE())
            rec.setCovA()
            rec.setCovE()        
            k += 1
    
        saveEpithelium(epith2, "1. SavedEpi_" + str(qspace.getSize()[0]) + purp)
        
        i += 1

    
def changeOne(name, dim, col, scale):
    """Given a file with (0,4) qspace, change some columns while keeping everything else constant. Return
    3 new saved epithelium files associated with 3 qspaces.
    Precondition: col in ["aff", "eff"] and scale is a 2D list with a range
    AND the "name" file AND other qspace files are in the correct directory"""
    assert col in ["aff", "eff"]
    epi = loadEpithelium(name + ".csv")
    sdSave = []

    for rec in epi.getRecs():
        i = 0
        sd = []
        while i < dim:
            sd.append(random.uniform(scale[0],scale[1]))
            i += 1
        if col == "aff":
            rec.setSdA(sd)
            rec.setCovA()
        else: #col == "eff":
            rec.setSdE(sd)
            rec.setCovE()
        sdSave.append(sd)
    
    name2 = name[:name.index("(")] + "(0, 10)"
    epi2 = loadEpithelium(name2 + ".csv")
    i=0
    for rec in epi2.getRecs():
        if col == "aff":
            rec.setSdA(sdSave[i])
            rec.setCovA()
        else:
            rec.setSdE(sdSave[i])
            rec.setCovE()
        i += 1
    
    name3 = name[:name.index("(")] + "(0, 30)"
    epi3 = loadEpithelium(name3 + ".csv")
    i=0
    for rec in epi3.getRecs():
        if col == "aff":
            rec.setSdA(sdSave[i])
            rec.setCovA()
        else:
            rec.setSdE(sdSave[i])
            rec.setCovE()
        i += 1

    saveEpithelium(epi, name +  ", " + col + "_sd=" + str(scale))
    saveEpithelium(epi2, name2 +  ", " + col + "_sd=" + str(scale))
    saveEpithelium(epi3, name3 +  ", " + col + "_sd=" + str(scale))

def changeMean(name, dim, scale):
    """Given a file with name, change mean columns to new qspace scale
    while keeping everything else constant. Return new saved epithelium file.
    Precondition: scale = two extremes of the new qspace
    and the "name" file is in the correct directory"""
    
    epi = loadEpithelium(name + ".csv")

    for rec in epi.getRecs():
        i = 0
        mean = []
        while i < dim:
            mean.append(random.uniform(scale[0],scale[1]))
            i += 1
            rec.setMean(mean)
    
    newName = "1. SavedEpi_(" + str(scale[0]) + ", " + str(scale[1]) + ")"
    saveEpithelium(epi, newName)
    
#########Below are three simulations for RecDensity vs DpsiBar graphs
#The first test varyling ligands from 200 to 50 and returns 4 graphs
#The second tests varying aff and eff


def testRecDensityDpsiGraph1():
    """Testing for varying odorscenes. Returns 4 graphs - one with 200 ligands in an odorscene
    all the way to 50 ligands in an odorscene. Graph is dist btwn recs vs dpsibar"""
    dim = 2
    qspace = QSpace([(0,2), (0, 2)])
    numOdo = 200
    odorscene = createOdorscene(dim, [1e-5], [numOdo], qspace)
    PDFname="receptor distance vs dPsi, varying ligands 2"

    while numOdo >= 50:
        labelName = str(numOdo) + " odors"
        excelName = "Rec dist vs Dpsi, " + str(numOdo) + " odorants 2"
        #numReceptors, dn, #ofLigands in Odorscene, dim, name, label name, fixed efficacy
        recDensityDpsiGraph(.01, qspace, odorscene, dim, PDFname, labelName, excelName, .5, False)
        numOdo -=50

def testRecDensityDpsiGraph2():
    """Testing for varying aff and eff standard deviations
    Returns 3 graphs - first with affSD=effSD=1 all the way to affSD=effSD=.25"""
    qspace = QSpace([(0,4), (0, 4)])
    numOdo = 100
    sd = 1
    dim = 2
    odorscene = createOdorscene(dim, [1e-5], [numOdo], qspace)
    
    PDFname="receptor distance vs dPsi, varying Standard Dev"
    
    while sd >= .25:
        labelName = str(sd) + " sd"
        excelName = "Rec dist vs Dpsi, " + str(sd) + " sd"
        #numReceptors, dn, #ofLigands in Odorscene, dim, name, label name, fixed efficacy
        recDensityDpsiGraph(.01, qspace, odorscene, dim, PDFname, labelName, excelName, sd, False)
        if sd == 1:
            sd -= .5
        else:
            sd-= .25

#####Histogram to show that model works when varying eff and aff
def effAnalysis(effSD, affSD=[2,2], qspace=(0,4), fixed=False):
    """Goal: Show that our model works - varying eff and aff creates agonists etc.    
    Returns one graph with a histogram of number of locations (there is a ligand at each location)
    that activate a receptor to a specific activation and a line graph
    of avg efficacy in each "activation section"
    **qspace argument is converted to actual QSpace in the function. Just input (x,y)
    Preconditions: effSD is the distribution scale for the SD [x1,x2]"""
    
    #Consants
    dim = 2
    qspace = QSpace([qspace, qspace])
    odorscenes = []  #Create 1600 odorscenes (with 1 ligand each) that span qspace from 0,0 to 3.9,3.9
    gl = layers.createGL(1)
    i = 0.0
    ID = 0
    while i < qspace.getSize()[0][1]:
        j = 0.0
        while j < qspace.getSize()[1][1]:
            odo = Ligand(ID, [i,j], 1e-5) #ID, Loc, Conc
            odorscenes.append(Odorscene(0,[odo]))
            ID += 1
            j += .1
        i += .1
    
    epi = createEpithelium(1, dim, qspace, affSD, effSD, True) #Creates an epithelium with 1 rec (and not constant mean)
    print "Aff sd distr: " + str(epi.getRecs()[0]._sdA)
    print "eff sd distr: " + str(epi.getRecs()[0]._sdE)
    print "mean is " +str(epi.getRecs()[0]._mean)
    
    bins = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]
    xAxis2 = [.05,.15,.25,.35,.45,.55,.65,.75,.85,.95]
    yAxis_act = [0]*10
    yAxis_eff = [0]*10

    #Loop through ligands, activate
    #Within loop:
    for odors in odorscenes:
        ActivateGL_QSpace(epi, odors, gl, fixed) #if fixed=True, eff is fixed at 1
        activ= epi.getRecs()[0].getActiv() 
        index = int(math.floor(activ*10.0))
        
        yAxis_act[index] += 1 #Add 1 to the correct location based on activation
        yAxis_eff[index] += odors.getOdors()[0]._eff

    i = 0
    for elem in yAxis_eff: #divide to get the avg efficacy
        #yAxis_eff[i] = elem/(float(len(odorscenes)))
        yAxis_eff[i] = elem/(float(yAxis_act[i]))
        i +=1

    print "activation bin: " + str(yAxis_act)
    print "mean efficacy: " + str(yAxis_eff)
    
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

def occVsLocGraph(affList=[2,1.5,1,.5]):
    """Takes a Receptor instance in a qspace=(0,4) and outputs
    Occupancy vs Location plot for rec with affSD=affList
    Highest affSD is a solid line on the graph"""
    dim = 1
    qspace = QSpace([(0,4)])
    odorscenes = []  #Create 1600 odorscenes (with 1 ligand each) that span qspace from 0 to 3.9
    gl = layers.createGL(1)
    i = 0.0
    ID = 0
    while i < qspace.getSize()[0][1]-.01:
        odo = Ligand(ID, [i], 1e-5) #ID, Loc, Conc
        odorscenes.append(odo)
        ID += 1
        i += .01

    recs = []
    for aff in affList:
        recs.append(Receptor(1,[2],[aff],[1])) #Id, mean, sda, sde
 
    index = 0
    for rec in recs:
        df = 0
        location=[]
        occupancy=[]
        labelName = "AffSD=" + str(rec.getSdA())
        if max(affList) in rec.getSdA():
            line = "-"
        else:
            line = "--"
        for odor in odorscenes:
            aff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovA())
            aff = aff / rec.getScale() #Scales it from 0 to 1
                
            #Now convert gaussian aff to kda
            aff = 10**((aff * (peak_affinity - minimum_affinity)) + minimum_affinity) ##peak_affinity etc. are global variables
            
            odor.setAff(float(aff))
            df = odor.getConc()/odor._aff
    
            location.append(odor.getLoc())
            occ = ( (1) / (1 + ( (odor._aff/odor.getConc()) * (1 + df - (odor.getConc() / odor._aff ) ) ) **m) ) #m=1
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
    
def effVsLocGraph(effList=[.1,.5,1,2,3]):
    """Takes a Receptor instance in a full qspace and outputs
    Efficacy vs Location plot for rec with effSD=effList.
    The highest effSD gives a solid line"""
    dim = 1
    qspace = QSpace([(0,4)])
    odorscenes = []  #Create 1600 odorscenes (with 1 ligand each) that span qspace from 0 to 3.9
    gl = layers.createGL(1)
    i = 0.0
    ID = 0
    while i < qspace.getSize()[0][1]-.01:
        odo = Ligand(ID, [i], 1e-5) #ID, Loc, Conc
        odorscenes.append(odo)
        ID += 1
        i += .01

    recs = []
    for eff in effList:
        recs.append(Receptor(1,[2],[1],[eff])) #Id, mean, sda, sde
        
    index = 0
    for rec in recs:
        df = 0
        location=[]
        efficacy=[]
        labelName = "EffSD=" + str(rec.getSdE())
        effScale = float(mvn.pdf(rec.getMean(), rec.getMean(), rec.getCovE())  )

        if max(effList) in rec.getSdE():
            line = "-"
        else:
            line = "--"
        for odor in odorscenes:
            eff = mvn.pdf(odor.getLoc(), rec.getMean(), rec.getCovE())
            eff = float(eff) / effScale #Scales it from 0 to 1
    
            location.append(odor.getLoc())
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

def runDPsiOccActGraphFromExcel(aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, purpose="standard"):
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, c, 2)
    
    nameDpsi1= "dPsi, qspace=(0, 4)" + purp + ".csv"
    nameDpsi2= "dPsi, qspace=(0, 10)" + purp + ".csv"
    nameDpsi3= "dPsi, qspace=(0, 30)" + purp + ".csv"
    nameAO1 = "LigandSat with (0, 4) qspace" + purp + ".csv"
    nameAO2 = "LigandSat with (0, 10) qspace" + purp + ".csv"
    nameAO3 = "LigandSat with (0, 30) qspace" + purp + ".csv"
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400]
    titleName = "DpsiBar vs Occ and Act" + purp
    pdfName = "DpsiBar vs Occ and Act" + purp
    
    dPsiOccActGraphFromExcel(nameDpsi1, nameAO1, xaxis, numRecs, "(0,4) qspace", titleName, pdfName, 'b', rep=200.0, close=False)
    dPsiOccActGraphFromExcel(nameDpsi2, nameAO2, xaxis, numRecs, "(0,10) qspace", titleName, pdfName, 'g', rep=200.0, close=False)
    dPsiOccActGraphFromExcel(nameDpsi3, nameAO3, xaxis, numRecs, "(0,30) qspace", titleName, pdfName, 'r', rep=200.0, close=True)
    
def purpFunction(purpose, aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2):
    """Returns a string that can be used for titles etc."""
    if purpose == "aff": 
        return ", aff_sd=" + str(aff_sd)
    elif purpose == "eff":
        if len(eff_sd) == 1:
            return ", eff_sd=" + str(eff_sd[0])
        else:
            return ", eff_sd=" + str(eff_sd)
    elif purpose == "c":
        return ", glom_pen=" + str(glom_penetrance)
    elif purpose == "recs":
        return ", numRecs=" + str(numRecs)
    elif purpose == "redAff":
        num = str(8 + peak_affinity)
        return ", redAff by 10^" + num
    elif purpose == "dim":
        return ", dim=" + str(dim)
    else: #purpose == "standard"
        return ""
    
def test():
    ####dPsiBarSaturation simulations

    testdPsiBarSaturation_Qspaces(fixed=False, aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4,10,30], purpose="standard")
    #testdPsiBarSaturationDim(dims=[3,4], fixed=False, aff_sd=[.5,1.5], eff_sd=[.05,1.0], numRecs=30, c=1)
    
    ####Creating dPsi vs Occ and Act graphs from excel docs
    
    #runDPsiOccActGraphFromExcel(aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, purpose="redAff")

    ####Creating similar epithelium files
    
    #changeMean("1. SavedEpi_(0, 4)", 2, [0,15])
    #changeOne("1. SavedEpi_(0, 4)", 2, "aff", [.5,2.5])
    #makeSimilar(30, [.5,1.5], [0.05,1.0], "dim", [4,10,30], 4)

    ####RecDensity vs Dpsi simulations:
    
    #testRecDensityDpsiGraph1()
    #testRecDensityDpsiGraph2()
    
    ####Histogram for paper - proving model works with regards to efficacy
    
    #effAnalysis([.5,1.0], affSD=[2,2], qspace=(0,4), fixed=True)
    
    ####Loc graphs vs Occ and Eff
    #occVsLocGraph(affList=[2,1.5,1,.5])
    #effVsLocGraph(effList=[3,2,1,.5,.1])
    
test()