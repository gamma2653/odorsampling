# Testing layers.py
# Mitchell Gronowitz
# Spring 2015

# Edited by Christopher De Jesus
# Summer 2023

from __future__ import annotations

from odorsampling.layers import (
    euclideanDistance, ActivateMCLfromGL, apply_sample_map,
    GraphGlomActivation, GraphMitralActivation, normalize, graphLayer, colorMapWeights,
    MitralLayer, GlomLayer
) 
import matplotlib.pyplot as plt



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
    gl = GlomLayer.create(1000)
    
    gl.activate_random(sel, mean, sd)
    #x-axis
    x = [0.0,.1, .2, .3, .4, .5, .6, .7, .8, .9]
    
    #Make the width .09 for each bar
    w = [0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09]
    
    #Creating y-axis
    y = [0,0,0,0,0,0,0,0,0,0]
    count = 0
    while count < len(gl):
        index = gl[count].activ
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
    layers = GlomLayer.create_array(gl, x, opt, sel, num, mean, sd)
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
    gl = GlomLayer.create(2000)
    gl.activate_random("u")
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
    gl = GlomLayer.create(5)
    mcl = MitralLayer.create(5)
    print('\n' + "Testing a fixed simple sampling map with cr=3" + '\n')
    Map = mcl.createSamplingMap(gl, 3, True, "simple")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))

    print('\n' + "Testing an unfixed simple sampling map with cr=3, sd=2" '\n')
    Map = mcl.createSamplingMap(gl, 3, False, "simple", 2, "lin")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    
    print('\n' + "Testing a fixed bias linear sampling map with cr=3" + '\n')
    Map = mcl.createSamplingMap(gl, 3, True, "bias", 2, "lin")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    
    print('\n' + "Testing an unfixed bias linear sampling map with cr=3, sd=2" + '\n')
    Map = mcl.createSamplingMap(gl, 3, False, "bias", 2, "lin")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
        
    print('\n' + "Testing a fixed bias exp sampling map with cr=3" + '\n')
    Map = mcl.createSamplingMap(gl, 3, True, "bias", 2, "exp")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
        
    print('\n' + "Testing an unfixed bias exp sampling map with cr=3, sd=2" + '\n')
    Map = mcl.createSamplingMap(gl, 3, False, "bias", 2, "exp")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))


def testCreateandLoadFile():
    """Tests createMCLSamplingMap() and loadMCLSamplingMap()"""

    print("Testing MCL store and load")
    gl = GlomLayer.create(10)
    mcl = MitralLayer.create(6)
    Map = mcl.createSamplingMap(gl, 4, True, "simple")
    ActivateMCLfromGL(gl, mcl, "add", Map)
    mcl.save("testmcl")
    mcl2 = MitralLayer.load("testmcl.mcl")
    print("mcl1:")
    for m in mcl:
        print(m)
    print("mcl2:")
    for m in mcl2:
        print(m)
    
    

def testApplyMCLSamplingMap():
    """Tests building the connections btwn MCL and GL (created with total number of gloms) given a map"""
    gl = GlomLayer.create(5)
    gl.activate_random("u")
    mcl = MitralLayer.create(5)
    Map = mcl.createSamplingMap(gl, 4, True, "simple")
    #print(Map)
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    apply_sample_map(gl,mcl,Map)
    for mitral in mcl:
        print(mitral)
    for glom in gl:
        print(str(glom) + " # of connections: " + str(glom.conn))
    print("items returns: " + str(mcl[0].glom.items()))
    print("items returns: " + str(mcl[1].glom.items()))
    print("items returns: " + str(mcl[2].glom.items()))
    print("items returns: " + str(mcl[3].glom.items()))
    print("items returns: " + str(mcl[4].glom.items()))
    

def testApplyMCLSamplingMapBalanced():
    """Tests building the connections btwn MCL and GL (created with dimensions) given a map using BALANCED RANDOM"""
    # FIXME: hotfix
    gl = GlomLayer.createGL_dimensions(4,2)
    gl.activate_random("u")
    mcl = MitralLayer.create(8)
    Map = mcl.createSamplingMap(gl, 4, True, "balanced")
    #print(Map)
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    apply_sample_map(gl,mcl,Map)
    for mitral in mcl:
        print(mitral)
        print(mitral.loc)
    for glom in gl:
        print(str(glom) + " # of connections: " + str(glom.conn))
        print(glom.loc)
    print("items returns: " + str(mcl[0].glom.items()))
    print("items returns: " + str(mcl[1].glom.items()))
    print("items returns: " + str(mcl[2].glom.items()))
    print("items returns: " + str(mcl[3].glom.items()))
    print("items returns: " + str(mcl[4].glom.items()))
    print("items returns: " + str(mcl[5].glom.items()))
    print("items returns: " + str(mcl[6].glom.items()))
    print("items returns: " + str(mcl[7].glom.items()))

def testApplyMCLSamplingMapLocation():
    """Tests building the connections btwn MCL and GL (created with dimensions) given a map using LOCATION"""
    # FIXME: hotfix
    gl = GlomLayer.createGL_dimensions(4,4)
    gl.activate_random("u")
    mcl = MitralLayer.create(4)
    Map = mcl.createSamplingMap(gl, 10, True, "location")
    #print(Map)
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    apply_sample_map(gl,mcl,Map)
    ActivateMCLfromGL(gl, mcl, "add", Map, "None")
    for mitral in mcl:
        print(mitral)
        print(mitral.loc)
    for glom in gl:
        print(str(glom) + " # of connections: " + str(glom.conn))
        print(glom.loc)
    print("items returns: " + str(mcl[0].glom.items()))
    print("items returns: " + str(mcl[1].glom.items()))
    print("items returns: " + str(mcl[2].glom.items()))
    print("items returns: " + str(mcl[3].glom.items()))
    # print("items returns: " + str(mcl[4].glom.items()))
    # print("items returns: " + str(mcl[5].glom.items()))
    # print("items returns: " + str(mcl[6].glom.items()))
    # print("items returns: " + str(mcl[7].glom.items()))
    print("\ndone")

def testGraphGlomActivation():
    """ """
    gl = GlomLayer.createGL_dimensions(10,10)
    gl.activate_random("u")
    GraphGlomActivation(gl, gl[0].dim[1], gl[0].dim[0])

def testGraphMitralActivation():
    """ """
    gl = GlomLayer.createGL_dimensions(4,4)
    gl.activate_random("u")
    mcl = MitralLayer.create(10)
    Map = mcl.createSamplingMap(gl, 10, True, "location")
    #print(Map)
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    apply_sample_map(gl,mcl,Map)
    ActivateMCLfromGL(gl, mcl, "add", Map, None)
    for mitral in mcl:
        print(mitral)
        print(mitral.loc)
    for glom in gl:
        print(str(glom) + " # of connections: " + str(glom.conn))
        print(glom.loc)
    
    GraphMitralActivation(gl, mcl, 4, 4)

def testActivateMCLfromGL():
    """Testing ActivateMCLfromGL"""
    gl = GlomLayer.create(6)
    gl.activate_random("u")
    mcl = MitralLayer.create(6)
    Map = mcl.createSamplingMap(gl, 3, True, "simple")
    for elem in Map:
        print("Mitral: " + str(elem[0]) + " Glom: " + str(elem[1]) + " Weight: " + str(elem[2]))
    print("\n testing add:" + '\n')
    ActivateMCLfromGL(gl, mcl, "add", Map, None)
    print("glom: ")
    for glom in gl:
        print(glom)
        # print("OH HELLO " + str(glom.loc[0]))
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
    mcl = MitralLayer.create(6)
    mcl[0].activ = 1.5
    mcl[1].activ = 1
    mcl[2].activ = 1.2
    mcl[3].activ = .3
    mcl[4].activ = .1
    mcl[5].activ = .5
    mcl = normalize(mcl)
    for m in mcl:
        print(m)


def testgraphLayer():
    """Testing graphLayer for GL then MCL"""
    gl = GlomLayer.create(20)
    gl.activate_random("u")
    mcl = MitralLayer.create(15)
    map_ = mcl.createSamplingMap(gl, 4, True, "simple")
    ActivateMCLfromGL(gl, mcl, "add", map_, "u", .05)
    st = ""
    ind = 0
    while ind < len(gl):
        st = st + str(ind) + ": " + str(gl[ind].activ) + " "
        ind += 1
    print(st)
    graphLayer(gl, False)
    graphLayer(mcl, True)
    
def testColorMap():
    gl = GlomLayer.create(4)
    gl.activate_random("u")
    mcl = MitralLayer.create(5)
    Map = mcl.createSamplingMap(gl, 2, True, "simple")
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