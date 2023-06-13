#Testing layers.py
#Mitchell Gronowitz
#Spring 2015

from layers import *
import cells
import random
import matplotlib.pyplot as plt
import math



#####Test 1: Random generation of glom activation levels (testing activateGL_Random())

#Graphical representations of above function
def testGraphGlomActivLvl(mean, sd):
    """Graphically represents the activation levels for all three distributions
    given the same glomeruli level."""
    _graphHelper("u",mean,sd)
    _graphHelper("g",mean,sd)
    _graphHelper("e",mean,sd)


def _graphHelper(sel, mean, sd):
    """Graphically represents the activation levels for distribution
    specified by sel with mean and sd."""
    gl = createGL(1000)
    
    activateGL_Random(gl, sel, mean, sd)
    #x-axis
    x = [0.0,.1, .2, .3, .4, .5, .6, .7, .8, .9]
    
    #Make the width .09 for each bar
    w = [0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09]
    
    #Creating y-axis
    y = [0,0,0,0,0,0,0,0,0,0]
    count = 0
    while count < len(gl):
        index = gl[count].getActiv()
        index = int(index*10)
        if index == 10:
            index == 9
        y[index] += 1
        count += 1
    plt.bar(x, y, w)
    if sel == "u":
        plt.title('Uniform distribution')
    elif sel == "g":
        plt.title("Gaussian Distribution")
    else:
        plt.title("Exponential Distribution")    
    
    plt.xlabel("Activation Level")
    plt.ylabel("# of Glom at given Activation Range")
    plt.show()


######Test 2: Generation of similar activation levels in a GL array (testing createGLArray())
#4 test cases: uniform star, uniform series, gaussian star, gaussian series

def testGraphGLArraySimilarity(gl, x, opt, sel, num=0, mean=0, sd=0 ):
    """Draws a graph of the euclidean distance between the activation levels
    of the glomeruli in gl and the activation levels in the generated similar GLArray.
    x=len(GLArray), opt = star/ser, sel = gaussian or uniform."""
    inc = 1
    y = []
    while inc <= x:
        y.append(inc)
        inc += 1
    layers = createGLArray(gl, x, opt, sel, num, mean, sd)
    inc = 0
    axis = []
    assert x == len(layers), "Worst."
    while inc < len(layers):
        axis.append(euclideanDistance(gl,layers[inc]))
        inc += 1
    assert len(axis) == len(y), "AHHHH"
    
    plt.plot(y, axis)
    if opt == "ser":
        plt.title("Similar Glomeruli Layer Activation Patterns: Series")
    else:
        plt.title("Similar Glomeruli Layer Activation Patterns: Star")
    plt.xlabel("Odor")
    plt.ylabel("Euclidean Distance from Original GL")
    plt.show()


def testSimilar():
    """Using above functions to test activateGL_Random function"""
    gl = createGL(2000)
    activateGL_Random(gl, "u")
    #Test series where incremented number was chosen uniformly
    testGraphGLArraySimilarity(gl, 100, "ser", "u", .01, mean=0, sd=0 )
    #Test series with gaussian
    testGraphGLArraySimilarity(gl, 100, "ser", "g", .01, mean=.1, sd=.01 )
    #Test star with uniform
    testGraphGLArraySimilarity(gl, 100, "star", "u", .01, mean=0, sd=0 )
    #Test stat with gaussian
    testGraphGLArraySimilarity(gl, 100, "star", "g", .01, mean=.1, sd=.01)


######Test 3: Testing Map building (Testing CreateMCLSamplingMap())

def testMapBuidling():
    """Tests createMCLSamplingMap()"""
    gl = createGL(5)
    mcl = createMCL(5)
    print('\n' + "Testing a fixed simple sampling map with cr=3" + '\n')
    Map = createMCLSamplingMap(gl, mcl, 3, True, "simple")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))

    print('\n' + "Testing an unfixed simple sampling map with cr=3, sd=2" '\n')
    Map = createMCLSamplingMap(gl, mcl, 3, False, "simple", 2, "lin")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    
    print('\n' + "Testing a fixed bias linear sampling map with cr=3" + '\n')
    Map = createMCLSamplingMap(gl, mcl, 3, True, "bias", 2, "lin")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    
    print('\n' + "Testing an unfixed bias linear sampling map with cr=3, sd=2" + '\n')
    Map = createMCLSamplingMap(gl, mcl, 3, False, "bias", 2, "lin")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
        
    print('\n' + "Testing a fixed bias exp sampling map with cr=3" + '\n')
    Map = createMCLSamplingMap(gl, mcl, 3, True, "bias", 2, "exp")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
        
    print('\n' + "Testing an unfixed bias exp sampling map with cr=3, sd=2" + '\n')
    Map = createMCLSamplingMap(gl, mcl, 3, False, "bias", 2, "exp")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))


def testCreateandLoadFile():
    """Tests createMCLSamplingMap() and loadMCLSamplingMap()"""
    
    #commented out so only MCL load and store is tested
    """print("Testing GL store and load")
    gl = createGL(5)
    activateGL_Random(gl, "u")
    saveGL(gl, "test")
    gl2 = loadGL("test.GL")
    inc = 0
    Bool = True
    while inc < len(gl):
        if gl[inc].getId() != gl2[inc].getId() or gl[inc].getActiv() != gl2[inc].getActiv():
            Bool = False
        if gl[inc].getConn() != gl2[inc].getConn() or gl[inc].getLoc()[0] != gl2[inc].getLoc()[0]:
            Bool = False
        inc+=1
    print(str(Bool))
    
    print("Testing Map store and load")
    gl = createGL(5)
    mcl = createMCL(3)
    Map = createMCLSamplingMap(gl, mcl, 3, True, "simple")
    saveMCLSamplingMap(Map, "test")
    Map2 = loadMCLSamplingMap("test.mapGLMCL")
    Same = True
    inc = 0
    while inc < len(Map):
        if Map[inc][0] != Map2[inc][0]:
            Same = False
            print("MCL not equal")
        if Map[inc][1] != Map2[inc][1]:
            Same = False
            print("GL not equal")
        ####Weights not checked due to rounding issues
        inc+=1
    print(str(Same))"""

    print("Testing MCL store and load")
    gl = createGL(10)
    mcl = createMCL(6)
    Map = createMCLSamplingMap(gl, mcl, 4, True, "simple")
    ActivateMCLfromGL(gl, mcl, "add", Map)
    saveMCL(mcl, "testmcl")
    mcl2 = loadMCL("testmcl.mcl")
    print("mcl1:")
    for m in mcl:
        print(m)
    print("mcl2:")
    for m in mcl2:
        print(m)
    
    

def testApplyMCLSamplingMap():
    """Tests building the connections btwn MCL and GL (created with total number of gloms) given a map"""
    gl = createGL(5)
    activateGL_Random(gl, "u")
    mcl = createMCL(5)
    Map = createMCLSamplingMap(gl, mcl, 4, True, "simple")
    #print(Map)
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    ApplyMCLSamplingMap(gl,mcl,Map)
    for mitral in mcl:
        print(mitral)
    for glom in gl:
        print(str(glom) + " # of connections: " + str(glom.getConn()))
    print("items returns: " + str(mcl[0].getGlom().items()))
    print("items returns: " + str(mcl[1].getGlom().items()))
    print("items returns: " + str(mcl[2].getGlom().items()))
    print("items returns: " + str(mcl[3].getGlom().items()))
    print("items returns: " + str(mcl[4].getGlom().items()))
    

def testApplyMCLSamplingMapBalanced():
    """Tests building the connections btwn MCL and GL (created with dimensions) given a map using BALANCED RANDOM"""
    gl = createGL_dimensions(4,2)
    activateGL_Random(gl, "u")
    mcl = createMCL(8)
    Map = createMCLSamplingMap(gl, mcl, 4, True, "balanced")
    #print(Map)
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    ApplyMCLSamplingMap(gl,mcl,Map)
    for mitral in mcl:
        print(mitral)
        print(mitral.getLoc())
    for glom in gl:
        print(str(glom) + " # of connections: " + str(glom.getConn()))
        print(glom.getLoc())
    print("items returns: " + str(mcl[0].getGlom().items()))
    print("items returns: " + str(mcl[1].getGlom().items()))
    print("items returns: " + str(mcl[2].getGlom().items()))
    print("items returns: " + str(mcl[3].getGlom().items()))
    print("items returns: " + str(mcl[4].getGlom().items()))
    print("items returns: " + str(mcl[5].getGlom().items()))
    print("items returns: " + str(mcl[6].getGlom().items()))
    print("items returns: " + str(mcl[7].getGlom().items()))

def testApplyMCLSamplingMapLocation():
    """Tests building the connections btwn MCL and GL (created with dimensions) given a map using LOCATION"""
    gl = createGL_dimensions(4,4)
    activateGL_Random(gl, "u")
    mcl = createMCL(4)
    Map = createMCLSamplingMap(gl, mcl, 10, True, "location")
    #print(Map)
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    ApplyMCLSamplingMap(gl,mcl,Map)
    ActivateMCLfromGL(gl, mcl, "add", Map, "None")
    for mitral in mcl:
        print(mitral)
        print(mitral.getLoc())
    for glom in gl:
        print(str(glom) + " # of connections: " + str(glom.getConn()))
        print(glom.getLoc())
    print("items returns: " + str(mcl[0].getGlom().items()))
    print("items returns: " + str(mcl[1].getGlom().items()))
    print("items returns: " + str(mcl[2].getGlom().items()))
    print("items returns: " + str(mcl[3].getGlom().items()))
    # print("items returns: " + str(mcl[4].getGlom().items()))
    # print("items returns: " + str(mcl[5].getGlom().items()))
    # print("items returns: " + str(mcl[6].getGlom().items()))
    # print("items returns: " + str(mcl[7].getGlom().items()))
    print("\ndone")

def testGraphGlomActivation():
    """ """
    gl = createGL_dimensions(10,10)
    activateGL_Random(gl, "u")
    GraphGlomActivation(gl, gl[0].getDim()[1], gl[0].getDim()[0])

def testGraphMitralActivation():
    """ """
    gl = createGL_dimensions(4,4)
    activateGL_Random(gl, "u")
    mcl = createMCL(10)
    Map = createMCLSamplingMap(gl, mcl, 10, True, "location")
    #print(Map)
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    ApplyMCLSamplingMap(gl,mcl,Map)
    ActivateMCLfromGL(gl, mcl, "add", Map, "None")
    for mitral in mcl:
        print(mitral)
        print(mitral.getLoc())
    for glom in gl:
        print(str(glom) + " # of connections: " + str(glom.getConn()))
        print(glom.getLoc())
    
    GraphMitralActivation(gl, mcl, 4, 4)

def testActivateMCLfromGL():
    """Testing ActivateMCLfromGL"""
    gl = createGL(6)
    activateGL_Random(gl, "u")
    mcl = createMCL(6)
    Map = createMCLSamplingMap(gl, mcl, 3, True, "simple")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    print("\n testing add:" + '\n')
    ActivateMCLfromGL(gl, mcl, "add", Map, "None")
    print("glom: ")
    for glom in gl:
        print(glom)
        # print("OH HELLO " + str(glom.getLoc()[0]))
    x = 0
    #print( '\n' + "map: ")
    #while x<4:
    #    print(Map[x])
    #    x += 1
    print('\n' + "mitral: ")
    for mitral in mcl:
        print(mitral)
    print("\n testing avg:")
    ActivateMCLfromGL(gl, mcl, "avg", Map, "u", .01,.01)
    print('\n' +"glom: ")
    for glom in gl:
        print(glom)
    x = 0
    #print('\n' + "map: ")
    #while x<4:
    #    print(Map[x])
    #    x += 1
    print('\n' + "mitral: ")
    for mitral in mcl:
        print(mitral)
    

def testNormalization():
    """Tests the normalize function"""
    mcl = createMCL(6)
    mcl[0]._activ = 1.5
    mcl[1]._activ = 1
    mcl[2]._activ = 1.2
    mcl[3]._activ = .3
    mcl[4]._activ = .1
    mcl[5]._activ = .5
    mcl = normalize(mcl)
    for m in mcl:
        print(m)


def testgraphLayer():
    """Testing graphLayer for GL then MCL"""
    gl = createGL(20)
    activateGL_Random(gl, "u")
    mcl = createMCL(15)
    Map = createMCLSamplingMap(gl, mcl, 4, True, "simple")
    ActivateMCLfromGL(gl, mcl, "add", Map, "u", .05)
    st = ""
    ind = 0
    while ind < len(gl):
        st = st + str(ind) + ": " + str(gl[ind].getActiv()) + " "
        ind += 1
    print(st)
    graphLayer(gl, False)
    graphLayer(mcl, True)
    
def testColorMap():
    gl = createGL(4)
    activateGL_Random(gl, "u")
    mcl = createMCL(5)
    Map = createMCLSamplingMap(gl, mcl, 2, True, "simple")
    ActivateMCLfromGL(gl, mcl, "add", Map, "u", .05)
    colorMapWeights(Map,gl,mcl)


def test():
    # testGraphGlomActivLvl(.5,.2)       #Testing assigning random activation levels to glomeruli
    #testSimilar()                     #Testing assigning similar activation levels to glom array
    # testMapBuidling()
    #testCreateandLoadFile()
    # testApplyMCLSamplingMap()
    # testApplyMCLSamplingMapBalanced()
    # testApplyMCLSamplingMapLocation()
    # testGraphGlomActivation()
    testGraphMitralActivation()
    # testActivateMCLfromGL()
    #testNormalization()
    #testgraphLayer()
    #testColorMap()

test()