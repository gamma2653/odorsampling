#Testing Ligands, Odorscenes, Receptors, and Epithelium (RnO) to ensure correct functionality
#Mitchell Gronowitz
#April and May 2015

from __future__ import annotations

from RnO import *
import random
import layers
import copy
from matplotlib.backends.backend_pdf import PdfPages
import config

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence


def testModifyLoc():
    """Testing modify location"""
    o1 = Ligand(0, [10, 4, 2], .000004)
    o2 = Ligand(1, [1, 2, 12], .000002)
    o3 = Ligand(2, [7, 0, 1], .000004)
    size =[(-5,5), (-4, 2), (-10, 10)]
    qspace = QSpace(size)
    
    modifyLoc(o1, qspace, 3)

    
def testOdorscene():
    """Testing odorscene objects"""
    size = QSpace([(-5,5), (-4, 2), (-10, 10)])
    od = createOdorscene(3, [.0004,.00003,.00001], [3,4,6], size)

    
def testCreateRec():
    """Testing createReceptor"""
    qspace = QSpace([(-5,5), (-4, 2), (-10, 10),(-5,5), (-4, 2)])
    rec = createReceptor(5, qspace)

def testCreateEpithelium():
    """Testing createEpithelium"""
    qspace = QSpace([(-.5,.5), (-.5, .5), (-.5,.5)])
    epi = createEpithelium(4, 3, qspace)

def testActivateGL_QSpace():
    """Testing activation of GL from epithelium and odorscene"""
    #Create a Glomeruli Layer with 3 Glomeruli with activation level = 0
    gl = layers.createGL(3)
    
    #Define a 2D Qspace 
    qspace = QSpace([(-.5,.5), (-.5, .5)])#Both dimensions are between -.5 and .5
    
    #Create an odorscene object with 5 diff ligands (2 with a conc of .0001 and 3 with a conc of .01)
    odorscene = createOdorscene(2, [.0001, .01], [3,5], qspace)  #dim,conc,amount,qspace
    
    #Create an epithelium object with 5 receptors (both are 2D)
    epith = createEpithelium(3, 2, qspace) #amt, dim    **amt = len(gl) and dim = dim of odorscene
    
    #Print out activation before calling activate function
    
    ActivateGL_QSpace(epith, odorscene, gl, False)

    #Print out activation after calling activate function

    
def testSaving():
    """Tests loading and saving objects as CSV files
    Uncomment whichever saving you want to test"""
    
    qspace = QSpace([(-2,2), (-1, 1), (-1,1), (-3,3)])
    
    ##Testing ligand
    #odor = Ligand(4, [3.5554, 5.333, 9.0], 1e-5)
    #saveLigand(odor, "testLigand")
    
    ##Testing odorscene
    #odorscene = createOdorscene(4, [.0001, .01], [2,3], qspace)  #dim,conc,amount,qspace
    #saveOdorscene(odorscene, "testOdorscene")
    
    ##Testing Receptor
    #rec = createReceptor(3, qspace)
    #print rec
    #saveReceptor(rec, "testRec")
    
    ##Testing Epithelium
    epi = createEpithelium(4,3,qspace)
    saveEpithelium(epi, "testEpi")


def testLoading():
    """After calling testSaving, this function loads the ligand that was just saved.
    Uncomment whichever saving you want to test"""
    ##Testing ligand
    #ligand = loadLigand("testLigand.csv")
    #print ligand
    
    ##Testing Odorscene
    #odorscene = loadOdorscene("testOdorscene.csv")
    #print odorscene
    
    ##Testing Receptor
    #rec = loadReceptor("testRec.csv")
    #print rec
    #print "aff is " + str(rec.getSdA())
    #print "eff is " + str(rec.getSdE())
    
    ##Testing Epithelium
    epi = loadEpithelium("testEpi.csv")


def testSumOfSquares():
    """Tests normal sumofSquares which creates a second similar odorscene to measure discriminability"""
    #Define a 3D Qspace 
    qspace = QSpace([(-.5,.5), (-.5, .5), (-.5,.5)])#Both dimensions are between -.5 and .5
    
    #Create an odorscene object with 4 diff ligands (all with .1 concentration)
    odorscene = createOdorscene(3, [.1], [4], qspace)  #dim,conc,amount,qspace
    
    #####Comparing psi value for very different receptors to very similar receptors
    #Create an epithelium object with 5 receptors (both are 2D)
    epith = createEpithelium(10, 3, qspace) #amt, dim    **amt = len(gl) and dim = dim of odorscene
    dn = [.6,.7,.8]
    diff = sumOfSquares(epith, odorscene,dn, True)
    
    dn = [.0001, .00001, .00003]
    diff = sumOfSquares(epith, odorscene,dn, True)
    
def testSumOfSquares2():
    """Tests sumofSquares2 which measures discriminability btwn two given odorscenes"""
    #Define a 3D Qspace 
    qspace = QSpace([(-.5,.5), (-.5, .5), (-.5,.5)])#Both dimensions are between -.5 and .5
    
    #Manually create 2 odorscenes:
    i = 0
    odors = []
    odors2 = []
    x = -.5
    y = -.4
    z = -.3
    a = -.45
    b = -.35
    c = -.25
    
    while i < 10:
        odors.append(Ligand(0,[x,y,z], 1e-5))
        odors2.append(Ligand(0,[a,b,c], 1e-5))
        x += .02
        y += .02
        z += .02
        a += .02
        b += .02
        c += .02
        i += 1
    epith = createEpithelium(30, 3, qspace) #amt, dim    **amt = len(gl) and dim = dim of odorscene
    diff = sumOfSquares2(epith, Odorscene(0,odors), Odorscene(0, odors2), True)
    
    diff = sumOfSquares(epith, Odorscene(0, odors), [.05,.05,.05], True)


def testSumofSquaresDetails():
    """Testing multiple things in sumofsquares"""
    #******Uncomment lines 805 and 806
    #epithelium, odorscene, dn, fixed=False
    qspace = QSpace([(0,4), (0, 4)])#Both dimensions are between 0 and 4 
    fixed = True
    conc = 1e-5
    
    epithelium = createEpithelium(30, 2, qspace)

    od1 = createOdorscene(2, [conc], [1], qspace)
    od2 = createOdorscene(2, [conc], [2], qspace)
    od5 = createOdorscene(2, [conc], [5], qspace)
    od10 = createOdorscene(2, [conc], [10], qspace)
    od20 = createOdorscene(2, [conc], [20], qspace)
    
    qspace_big = QSpace([(0,10), (0, 10)])
    od1big = createOdorscene(2, [conc], [1], qspace_big)
    od10big = createOdorscene(2, [conc], [10], qspace_big)
    epithelium_big = createEpithelium(30, 2, qspace_big)

    
    #print "comparing different num of ligands"
    #print "od1: " + str(sumOfSquares(epithelium, od1, [.01, .01], fixed))
    #print "od2: " + str(sumOfSquares(epithelium, od2, [.01, .01], fixed))
    #print "od5: " + str(sumOfSquares(epithelium, od5, [.01, .01], fixed))
    #print "od10: " + str(sumOfSquares(epithelium, od10, [.01, .01], fixed))
    #print "od20: " + str(sumOfSquares(epithelium, od20, [.01, .01], fixed))
    
    
    #print "Fixed efficacy vs unfixed"
    #print "od1 fixed: " + str(sumOfSquares(epithelium, od1, [.01, .01], True))
    #print "od1 unfixed: " + str(sumOfSquares(epithelium, od1, [.01, .01], False))


def increasingRecDistTest():
    """Trying to figure out why dPsi increases with decreasing density"""
    small_qspace = QSpace([(0, 3), (0, 3)])
    big_qspace = QSpace([(0, 7), (0, 10)])
    temp = QSpace([(0, .1), (0, .1)])
    

    epibig = createEpithelium(10, 2, big_qspace, [.5,.5])
    epismall = createEpithelium(10, 2, small_qspace, [.5,.5])
    
    odobig = createOdorscene(2, [1e-5], [10], big_qspace)
    odocopy = copy.deepcopy(odobig.odors)
    
    ligands = []
    for odo in odocopy:
        ligands.append(modifyLoc(odo, small_qspace, 2))
    odosmall = Odorscene(0, ligands)
    

def testdPsiBarCalc():
    #Define a 3D Qspace 
    qspace = QSpace([(-.5,.5), (-.5, .5), (-.5,.5)])#Both dimensions are between -.5 and .5
    
    #Create an odorscene object with 5 diff ligands (2 with a conc of .0001 and 3 with a conc of .01)
    odorscene = createOdorscene(3, [.0001, .01], [2,3], qspace)  #dim,conc,amount,qspace
    
    #Create an epithelium object with 5 receptors (both are 3D)
    epith = createEpithelium(3, 3, qspace) #amt, dim    **amt = len(gl) and dim = dim of odorscene
    dPsibar = dPsiBarCalcDiag(epith, odorscene, 1)
    dPsibar2 = dPsiBarCalcAngles(epith, odorscene, 1)

def testMultipleLigands():
    """Testing dPsiCalc for multiple ligands"""
    r = .01
    
    fixed = True
    smallQspace = QSpace([(0, 4), (0, 4)])#Both dimensions are between 0 and 4 
    
    epithelium = createEpithelium(30, 2, smallQspace)
    
    od1 = createOdorscene(2, [.04], [1], smallQspace)
    od2 = createOdorscene(2, [.04], [2], smallQspace)
    od10 = createOdorscene(2, [.04], [10], smallQspace)
    od25 = createOdorscene(2, [.04], [25], smallQspace)
    
    bigQspace = QSpace([(0,10), (0, 10)])
    
    epith2 = createEpithelium(30, 2, bigQspace)
    
    odo1 = createOdorscene(2, [.04], [1], bigQspace)
    odo2 = createOdorscene(2, [.04], [2], bigQspace)
    odo10 = createOdorscene(2, [.04], [10], bigQspace)
    odo25 = createOdorscene(2, [.04], [25], bigQspace)

def testIdentical():
    """In fixed eff, occ=act since eff=1. Therefore, if I run the same
    code, just one is fixed and one is not fixed, then the occ of the non-fixed
    will equal the occ and act columns of the fixed one.
    Let's make sure this is true."""

    qspace = QSpace([(0,4), (0, 4)])#Both dimensions are between 0 and 4
    epith = createEpithelium(30, 2, qspace, scale=[.5,1.5]) #amt, dim    **amt = len(gl) and dim = dim of odorscene
    pdfName = "LigandSat fixed vs not fixed"
    
    labelName = "not fixed"
    excelName = "LigandSat not fixed"
    #epi, dn, qspace, pdfName, labelName, excelName, fixed eff
    dPsiBarSaturation(epith, .01, qspace, pdfName, labelName, excelName, False) 
    
    labelName = "fixed"
    excelName = "LigandSat fixed"
    #epi, dn, qspace, pdfName, labelName, excelName, fixed eff
    dPsiBarSaturation(epith, .01, qspace, pdfName, labelName, excelName, True)


def testColorMapSumOfSquares():
    #Define a 2D Qspace 
    qspace = QSpace([(0,4), (0, 4)])#Both dimensions are between 0 and 4
    
    #Create an odorscene object with 25 ligands from 0,0 to 4,4
    # x = 0
    # y = 0
    # x = 1/(2*config.PIXEL_PER_Q_UNIT)
    # y = 1/(2*config.PIXEL_PER_Q_UNIT)
    y = 0
    x = 0
    ID = 0
    odorscenes = []
    while x < 80:
        y = 0
        # y = 1/(2*config.PIXEL_PER_Q_UNIT)
        while y < 80:
            # odorscenes.append(Odorscene(x,[Ligand(ID, [2.0*x/3.0 + 2.0/6.0, 2.0*y/3.0 + 2.0/6.0], .004)]))
            odorscenes.append(Odorscene(x,[Ligand(ID, [x/20.0, y/20.0], .004)]))

            #print odorscenes[y].
            y += 1
            ID += 1
        x += 1
    
    #Create an epithelium object with 5 receptors (both are 2D)
    epith = createEpithelium(2, 2, qspace)

    #Using mock receptors
    # recs= []
    # recs.append(Receptor(0, config.MOCK_RECEPTOR_MEAN, config.MOCK_RECEPTOR_SDA, config.MOCK_RECEPTOR_SDE))
    # recs.append(Receptor(1, config.MOCK_RECEPTOR_MEAN1, config.MOCK_RECEPTOR_SDA1, config.MOCK_RECEPTOR_SDE1))
    # recs.append(Receptor(2, config.MOCK_RECEPTOR_MEAN2, config.MOCK_RECEPTOR_SDA2, config.MOCK_RECEPTOR_SDE2))

    # epith = Epithelium(recs)


    #print odorscenes[440].getOdors()[0].getLoc()
    
    colorMapSumOfSquares(epith, odorscenes, .3, qspace)
    
def testSequentialOdorscene():
    qspace = QSpace([[0.0,10.0],[0.0,10.0]]) 
    odorscenes = sequentialOdorscenes(50, 10, 2, .17, qspace)
    #for odorscene in odorscenes:
    #    print odorscene
    epi = createEpithelium(100, 2, qspace, scale=[.05,1.0])
    simDpsi = sumOfSquares2(epi, odorscenes[0], odorscenes[1])
    difDpsi = sumOfSquares2(epi, odorscenes[0], odorscenes[49])
    
    
def testdPsiBarSaturation():
    #Define a 2D Qspace 
    qspace = QSpace([(0,4), (0, 4)])#Both dimensions are between 0 and 4
    c=9
    fixedEff = False
    
    epith = createEpithelium(30, 2, qspace, scale=[.5,1.5], scaleEff=[.05,1.0]) #amt, dim    **amt = len(gl) and dim = dim of odorscene
    pdfName = "LigandSat with " + str(qspace.size[0]) + " qspace"
    labelName = str(qspace.size[0]) + " qspace"
    excelName = pdfName
    plotTitle = "dPsiBarSaturation"
     #epi, dn, qspace, pdfName, labelName, excelName, fixed eff, plotTitle, Close
    dPsiBarSaturation(epith, .01, qspace, pdfName, labelName, excelName, fixedEff, c, plotTitle, True)

def testGraphFromExcel(toggle):
    """Returns "act" or "occ" graph vs # of ligands depending on toggle.
    Before running:
    1.Must move specific excel docs into research folder
    2.Input their exact titles.csv into name
    3.Change any titles and label names as necessary
    4.Be aware of how dPsiSaturation was run to develop the csv files - might need to
    change numRecs, xaxis, or rep in the function in RnO

    Precondition: toggle = "Act" or "Occ" """
    assert toggle in ["Act", "Occ"], "toggle is not act or occ"
    
    rep = 10.0
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400]
    numRecs = 30
    pdfName = "Rec" + toggle + " vs num of LigandsAHHH"
    titleName = "Rec " + toggle + " vs num of Ligands"
    
    name = "LigandSat with (0,4) qspace.csv"
    labelName = "(0,4) qspace"
    graphFromExcel(name, xaxis, numRecs, labelName, titleName, pdfName, toggle, rep, False)
    
    name = "LigandSat with (0,10) qspace.csv"
    labelName = "(0,10) qspace"
    graphFromExcel(name, xaxis, numRecs, labelName, titleName, pdfName, toggle, rep, False)
    
    name = "LigandSat with (0,30) qspace.csv"
    labelName = "(0,30) qspace"
    graphFromExcel(name, xaxis, numRecs, labelName, titleName, pdfName, toggle, rep, True)

def testRecDensityDpsiGraph():
    """Testing rec density vs discriminability given two diff odorscenes with diff # of ligands.
    returns two graphs - one for each odorscene."""
    dim = 2
    qspace = QSpace([(0,4), (0, 4)])
    numOdo = 100
    odorscene = createOdorscene(dim, [1e-5], [numOdo], qspace)
    PDFname="receptor distance vs dPsi, varying ligands"
    labelName = str(numOdo) + " odors"
    excelName = "Rec dist vs Dpsi, " + str(numOdo) + " odorants"
    
    #numReceptors, dn, #ofLigands in Odorscene, dim, name, labelName, excel name, fixed efficacy
    recDensityDpsiGraph(.01, qspace, odorscene, dim, PDFname, labelName, excelName, .5, False)
    
    numOdo = 50
    labelName = str(numOdo) + " odors"
    excelName = "Rec dist vs Dpsi, " + str(numOdo) + " odorants"
    recDensityDpsiGraph(.01, qspace, odorscene, dim, PDFname, labelName, excelName, .5, False)
    #***If each graph has the name, then graph 2 will override graph 1

def testRecInQspace():
    qspace = QSpace([(0, 4), (0, 4)])
    #uncomment print statements in recInQspace
    recInQspace(35**2, 2, qspace)
    
def testRecDensityDpsiGraphRandomized():
    r = .01
    dim = 2
    qspace = QSpace([(0, 4), (0, 4)])
    odorscene = createOdorscene(dim, [1e-5], [100], qspace)
    recDensityDpsiGraphRandomized(r, qspace, odorscene, dim, "TESTING", fixed=False)

def testGetLocations():
    """Tests getLocations which is a helper function for the new glomRecConn function"""
    pass
def testGlomRecConnNew():
    """tests new glomRecConn function. Ensures that running the same function
    on two different unactivated GLs with two similar odorscenes will produce 
    similar activation patterns"""
    qspace = QSpace([(0, 4), (0, 4)])
    
    glomLayer = layers.createGL(30)
    #odorscene = createOdorscene(2, [1e-5], [30], qspace, Id = 0)
    odorscene = sequentialOdorscenes(2, 30, 2, .01, qspace)
    
    epi = loadEpithelium("1. SavedEpi_(0, 4).csv")
    ActivateGL_QSpace(epi, odorscene[0], glomLayer, fixed=True, c=1, sel="avg")
    
    epi2 = loadEpithelium("1. SavedEpi_(0, 4).csv")
    ActivateGL_QSpace(epi2, odorscene[1], glomLayer, fixed=True, c=1, sel="avg")
    
    gl = layers.createGL(30)
    gl2 = copy.deepcopy(gl)
    
    conn = glomRecConnNew(epi.recs, gl, c=9, conn = [])
    glomRecConnNew(epi2.recs, gl2, c=9, conn=conn)
    

def testGlomRecConnNew2():
    """Prints out each glom's receptor connections and the associated weights"""
    qspace = QSpace([(0, 4), (0, 4)])
    
    epi = loadEpithelium("1. SavedEpi_(0, 4).csv")

    glomLayer = layers.createGL(30)
    odorscene = createOdorscene(2, [1e-5], [30], qspace, Id = 0)
    
    ActivateGL_QSpace(epi, odorscene, glomLayer, fixed=True, c=1, sel="avg")
    
    gl = layers.createGL(30)
    
    glomRecConnNew(epi.recs, gl)
    

        
def testGlomRecConnNew3():

    qspace = QSpace([(0, 4), (0, 4)])
    c=9
    numRecs = 100

    glomLayer = layers.createGL(numRecs)
    odorscene = sequentialOdorscenes(2, 5, 2, .01, qspace)
    
    epi = createEpithelium(numRecs, 2, qspace, [.5,1.5], [.05,1.0])    
    ActivateGL_QSpace(epi, odorscene[0], glomLayer, fixed=True, c=1, sel="avg")
    saveEpithelium(epi, "1. SavedEpi_(0, 4), 100 recs")
    
    epi2 = loadEpithelium("1. SavedEpi_(0, 4), 100 recs.csv")
    ActivateGL_QSpace(epi2, odorscene[1], glomLayer, fixed=True, c=1, sel="avg")
    
    gl = layers.createGL(numRecs)
    gl2 = copy.deepcopy(gl)
    
    conn = glomRecConnNew(epi.recs, gl, c)
    glomRecConnNew(epi2.recs, gl2, c, conn)
    
    i=0
    dpsi = 0
    while i < 3:
        dphi=0

        #print gl[i]._activ
        #print gl2[i]._activ
        dphi = gl2[i]._activ - gl[i]._activ
        dpsi += dphi**2
        i+=1

def testDPsiGraphFromExcel():
    name1 = "dPsi, qspace=(0, 4), glom_pen=0.68.csv"
    name2 = "dPsi, qspace=(0, 10), glom_pen=0.68.csv"
    name3 = "dPsi, qspace=(0, 30), glom_pen=0.68.csv"
    qspace1 = QSpace([(0, 4), (0, 4)])
    qspace2 = QSpace([(0, 10), (0, 10)])
    qspace3 = QSpace([(0, 30), (0, 30)])
    titleName = "testing"
    pdfName = "test"
    
    dPsiGraphFromExcel(name1, qspace1, titleName, pdfName, close=False)
    dPsiGraphFromExcel(name2, qspace2, titleName, pdfName, close=False)
    dPsiGraphFromExcel(name3, qspace3, titleName, pdfName, close=True)

def testDPsiOccActGraphFromExcel():
    #dPsiOccActGraphFromExcel(nameDpsi, nameAO, xaxis, numRecs, labelName, titleName, pdfName, rep=200.0, close=False)
    nameDpsi1= "dPsi, qspace=(0, 4).csv"
    nameDpsi2= "dPsi, qspace=(0, 10).csv"
    nameDpsi3= "dPsi, qspace=(0, 30).csv"
    nameAO1 = "LigandSat with (0, 4) qspace.csv"
    nameAO2 = "LigandSat with (0, 10) qspace.csv"
    nameAO3 = "LigandSat with (0, 30) qspace.csv"
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400]
    numRecs = 30
    titleName = "DpsiBar vs Occ and Act" #+ purp
    pdfName = "DpsiBar vs Occ and Act" #+ purp
    
    dPsiOccActGraphFromExcel(nameDpsi1, nameAO1, xaxis, numRecs, "(0,4) qspace", titleName, pdfName, 'b', rep=200.0, close=False)
    dPsiOccActGraphFromExcel(nameDpsi2, nameAO2, xaxis, numRecs, "(0,10) qspace", titleName, pdfName, 'g', rep=200.0, close=False)
    dPsiOccActGraphFromExcel(nameDpsi3, nameAO3, xaxis, numRecs, "(0,30) qspace", titleName, pdfName, 'r', rep=200.0, close=True)

    
        

def test():
    #####Testing simple functions

    #testModifyLoc()
    #testOdorscene()
    #testCreateRec()
    #testCreateEpithelium()
    #testActivateGL_QSpace()
    #testSaving()
    #testLoading()
    
    #####Testing "calculation" functions
    
    #testSumOfSquares()
    #testSumOfSquares2()
    #testSumofSquaresDetails()
    #increasingRecDistTest()
    #testdPsiBarCalc()
    #testMultipleLigands()
    #testIdentical()
    
    #####Testing simulations
    
    testColorMapSumOfSquares()
    #testSequentialOdorscene()
    #testdPsiBarSaturation()
    #testGraphFromExcel("Act")
    #testDPsiGraphFromExcel()
    #testDPsiOccActGraphFromExcel()

    #testRecDensityDpsiGraph()

    #testRecInQspace()
    #testRecDensityDpsiGraphRandomized()

    #testGetLocations()
    #testGlomRecConnNew()
    # testGlomRecConnNew2()
    #testGlomRecConnNew3()


test()
