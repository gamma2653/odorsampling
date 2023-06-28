#Glomeruli and Mitral Layers and related functions
#Mitchell Gronowitz
#Spring 2015

"""This module builds layers of Golemruli and Mitral cells, including connections between them.
This includes:
Building glom layer and initializing activation levels
Building a list of similarly activated glom layers given a gl
Building mitral layer
Building connections and initializing mitral activation levels
Saving GL, MCL, or Maps
Euclidean distance between two layers
Graphical representations of a layer
"""

import cells
import random
import os
import matplotlib.pyplot as plt
import math
import matplotlib.pylab
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def createGL(x):
    """Returns an array of x number of glom objects with activation levels
    and loc set to defaults. ID refers to its index in the array.
    Precondition: x is an int or float"""
    assert type(x) in [int, float], "x is not a number"
    GL = []
    count = 0
    while count<x:
        GL.append(cells.Glom(count, 0.0, [0,0], [0,0], 0))
        count += 1

    # for glom in GL:
    #     print str(glom)

    return GL

def createGL_dimensions(x, y):
    """Returns an array of x number of glom objects with activation levels
    and loc set to defaults. ID refers to its index in the array.
    Precondition: x is an int or float"""
    assert type(x) in [int, float], "x is not a number"
    assert type(y) in [int, float], "y is not a number"
    GL = []
    countX = 0
    countY = 0
    glomID = 0
    while countX<x:
        while countY<y:
            GL.append(cells.Glom(glomID, 0.0, [countX, countY], [y, x], 0))
            glomID += 1
            countY += 1
        countY = 0
        countX += 1

    return GL

def clearGLactiv(gl):
    """Given gl, clears all the activation lvls to 0.0"""
    for glom in gl:
        glom._activ = 0.0
        glom._recConn = {}

#The following function generates activation levels for a GL in different ways

#For now, if a number is generated to be over 1 or under 0 in Gaussian or
#exponential, the function will be called again to generate a different number.
def activateGL_Random(GL, sel, mean=0, sd=0):
    """Initializes activation level for given GL (glom layer).
    If sel = "u" then activation levels are drawn from a random distribution.
    If sel = "g" then drawn from a Gaussian distribution with mean and sd.
    If sel = "e", then drawn from exponenial distribution with mean.
    Precondition: GL is a list of Glom and sel is u, g, or e."""
    assert type(GL) == list, "GL is not a list"
    assert sel in ["g","u", "e"], "sel isn't g or u"
    assert (mean+sd) <= 1 and mean-sd >= 0, "Mean and SD are too high or low"
    for glom in GL:
        if sel == "u":
            x = random.random()
        elif sel == "g":
            x = random.gauss(mean, sd)
            while x > 1 or x < 0:
                x = random.gauss(mean, sd)
        else: #sel == "e":
            x = random.expovariate(1/mean)
            #Maybe find a different function that'll be able input as a parameter the rate of decay
            while x > 1 or x < 0:
                x = random.expovariate(1/mean)
        glom.setActiv(x)


#Creating array of GL's with similar activation
def createGLArray(gl, x, opt, sel, num, mean=0, sd=0):
    """Given a glomeruli layer, returns x amount of similar gl's using
    the similarity method specified with opt and sel. Original activ lvl is incremented by <= num.
    Preconditions: gl is a list of glomeruli, x is an int sel is star or ser"""
    assert type(x) == int, "x is not an int"
    assert  opt in ["star","ser"], "opt needs to be 'star' or 'ser'"
    #Everything else is asserted in helper function below
    if x == 0:
        return []
    if opt == "star":
        glom = activateGL_Similar(gl, num, sel, mean, sd)
        return [glom] + createGLArray(gl, x-1, opt, sel, num, mean, sd)
    else: #opt == "ser"
        glom = activateGL_Similar(gl, num, sel, mean, sd)
        return [glom] + createGLArray(glom, x-1, opt, sel, num, mean, sd)

#For now, any number that is incremented to be over 1 or under 0 is just set
# to 1 or 0 respectively.
def activateGL_Similar(gl, num, sel, mean=0, sd= 0, gl2=[]):
    """Returns a glomeruli layer with activation levels similar to gl, by randomly
    picking a number between -num and num and incrementing (or if gaussian then picked
    using mean and sd). If gl2 is empty, then create new gl, otherwise this function
    fills gl2 with activation levels.
    preconditions: len(gl2) == 0 or len(gl), num is between 0 and 1, sel is
    'g' for gaussian or 'u' for uniform"""
    assert gl2 ==[] or len(gl2) == len(gl), "Glomeruli layers are different sizes!"
    assert num > 0 and num < 1, "num must be between 0 and 1"
    assert sel in ["g", "u"], "sel is not 'g' or 'u'."
    if gl2 == []:
        gl2 = createGL(len(gl))
    for index in range(len(gl2)):
        if sel == "u":
            rand = random.uniform(-num,num)
        else: # sel == "g"
            rand = random.choice([1,-1])*random.gauss(mean, sd)
        inc = rand + gl[index].activ
        if inc > 1:
            inc = 1.0
        if inc < 0:
            inc = 0.0
        gl2[index].setActiv(inc)
    return gl2


def createMCL(x):
    """Returns an array of x number of mitral objects with activation levels,
    loc, and connecting glomeruli set to defaults. ID refers to its index in the array.
    Precondition: x is an int or float"""
    assert type(x) in [int, float], "x is not a number"
    ML = []
    count = 0
    while count<x:
        ML.append(cells.Mitral(count, 0.0, [0,0], {}))
        count += 1

    return ML

######## Creating Map

#****For now all weights are chosen uniformly
#connections are either fixed or cr serves as the mean with a sd for amount of connections
def createMCLSamplingMap(gl, mcl, cr, fix, sel, sd=0, bias="lin"):
    """Returns a map in the form of [[Mi, G, W],[Mi, G, W]...] where
    Mi and G are ID's and W is the weight.
    1. The convergence ratio (cr) determines the amount of glom sample to each
    mitral cell. Fix determines whether the cr is a fixed number or just the mean
    with sd.
    2. Sampling can be done randomly or biased (decided by sel).
    3. If biased, sampling can be done with linear bias or exponential.
    There may also be a cleanup function at the end.
    preconditions: fix is a boolean, cr is an int or float less than len(gl),
    bias is "lin" or "exp", and sel is "bias", "simple", "balanced", or "location"."""
    assert type(fix) == bool, "Fix is not a boolean"
    assert sel in ["bias", "simple", "balanced", "location"], "Sel is not a specific type!"
    assert type(cr) in [int, float] and cr + sd <= len(gl), "cr isn't a valid number"
    assert bias in ["lin", "exp"]
    if cr == 1 and fix == True:  #Needed to avoid two Mitral cells sampling the same Glom
        Map = oneToOneSample(gl,mcl)
    elif sel == "simple":
        Map = simpleSampleRandom(gl, mcl, cr, fix, sd)
    elif sel == "balanced": 
        assert (len(mcl)*cr)%len(gl) == 0, "cannot balance. recreate mitrals."
        Map = simpleSampleBalanced(gl, mcl, cr, fix, sd)
    elif sel == "location":
        assert gl[0].dim[0]>=3 and gl[0].dim[1]>=3, "glom need to be at least 3X3. recreate gloms."
        Map = simpleSampleLocation(gl, mcl, cr, fix, sd)
    else: #sel == "biased
        Map = biasSample(gl, mcl, cr, fix, bias, sd)
    #Can call a clean up function here if we want
    #print unsampledGlom(gl, mcl, Map)               #PRINTING HERE
    return Map

def oneToOneSample(gl,mcl):
    """For 1:1 sampling, each mitral cell chooses an unselected glom cell
    Precondition: len of gl >= len of mcl"""
    assert len(gl) >= len(mcl)
    Map = []
    indexes = range(0,len(gl))
    for mitral in mcl:
        ind = random.choice(indexes)
        indexes.remove(ind)
        Map.append([mitral.id, ind, 1])   #******Changed for weights to always be 1
    return Map

def simpleSample(gl, mcl, cr, fix, sd=0):
    """Builds a map by randomly choosing glomeruli to sample to mitral cells.
    If fix != true, cr serves as mean for # of glom sample to each mitral cell.
    Weights are randomly chosen uniformly.
    ***Weights of a mitral cell's gloms do NOT add up to 1.0***"""
    Map = []
    counter = 0
    if fix:
        while counter < len(mcl):
            inc = 0
            conn = []
            while inc < cr:
                num = random.randint(0,len(gl)-1)
                num = _prevDuplicates(num, conn, gl, "simple", [], 0) #Ensures that glom at num isn't connected to the mitral cell yet
                Map.append([mcl[counter].id, num, random.uniform(0,.4)])
                inc += 1
                conn.append(num)
            counter += 1
    if not fix:
        while counter < len(mcl):
            rand = max(random.gauss(cr, sd), 1)
            inc = 0
            conn = []
            while inc < rand:
                num = random.randint(0,len(gl)-1)
                num = _prevDuplicates(num, conn, gl, "simple", [], 0)
                Map.append([mcl[counter].id, num, random.uniform(0,.4)])
                inc += 1
                conn.append(num)
            counter += 1
    return Map

def simpleSampleRandom(gl, mcl, cr, fix, sd=0):
    """Builds a map by randomly choosing glomeruli to sample to mitral cells.
    If fix != true, cr serves as mean for # of glom sample to each mitral cell.
    Weights are randomly chosen uniformly.
    ***Weights of a mitral cell's gloms add up to 1.0***"""
    Map = []
    counter = 0
    if fix:
        while counter < len(mcl):
            inc = 0
            conn = []
            leftover = 1
            while inc < cr:
                num = random.randint(0,len(gl)-1)
                num = _prevDuplicates(num, conn, gl, "simple", [], 0) #Ensures that glom at num isn't connected to the mitral cell yet

                if inc == (cr-1):
                    act = leftover
                else:
                    act = random.uniform(0, leftover)
                    leftover -= act
                Map.append([mcl[counter].id, num, act])
                inc += 1
                conn.append(num)
            counter += 1
    if not fix:
        while counter < len(mcl):
            rand = max(random.gauss(cr, sd), 1)
            inc = 0
            conn = []
            while inc < rand:
                num = random.randint(0,len(gl)-1)
                num = _prevDuplicates(num, conn, gl, "simple", [], 0)
                Map.append([mcl[counter].id, num, random.uniform(0,.4)])
                inc += 1
                conn.append(num)
            counter += 1
    return Map

def simpleSampleBalanced(gl, mcl, cr, fix, sd=0):
    """Builds a map by randomly choosing glomeruli to sample to mitral cells.
    If fix != true, cr serves as mean for # of glom sample to each mitral cell.
    Weights are randomly chosen uniformly. Limits number of mitral cells that 
    glom can project to (Fanout_ratio = (#MC * cr) / #Glom).
    ***Weights of a mitral cell's gloms add up to 1.0***"""
    Map = []
    counter = 0
    F = (len(mcl) * cr)/len(gl) #fanout ratio
    glomSelections = []
    for g in gl:
        fanout = 0
        while fanout < F:
            glomSelections.append(g.id)
            fanout += 1

    if fix:
        while counter < len(mcl):
            inc = 0
            conn = []
            leftover = 1
            while inc < cr:
                num = random.choice(glomSelections)
                check = 0
                while num in conn and check < 100:
                    num = random.choice(glomSelections)
                    check += 1
                    assert check != 100, "please run again. this is check: " + str(check) # IMPLEMENT THIS IN A BETTER WAY. This error shows up when the very last mitral is forced to sample from the same glom.
                glomSelections.remove(num)
                if inc == (cr-1):
                    act = leftover
                else:
                    act = random.uniform(0, leftover)
                    leftover -= act
                Map.append([mcl[counter].id, num, act])
                inc += 1
                conn.append(num)
            counter += 1
    # currently we have decided that cr should always be fixed
    # if not fix:
        # while counter < len(mcl):
        #     rand = max(random.gauss(cr, sd), 1)
        #     inc = 0
        #     conn = []
        #     while inc < rand:
        #         num = random.randint(0,len(gl)-1)
        #         num = _prevDuplicates(num, conn, gl, "simple", [], 0)
        #         Map.append([mcl[counter].id, num, random.uniform(0,.4)])
        #         inc += 1
        #         conn.append(num)
        #     counter += 1
    return Map

def simpleSampleLocation(gl, mcl, cr, fix, sd=0):
    """Builds a map by randomly choosing glomeruli to sample to mitral cells.
    If fix != true, cr serves as mean for # of glom sample to each mitral cell.
    Weights are randomly chosen uniformly. Glomeruli are drawn randomly from the
    surrounding glomeruli that surround the parent glomerulus.
    ***Weights of a mitral cell's gloms add up to 1.0***"""
    Map = []
    counter = 0
    
    numLayers = math.ceil((-4+math.sqrt(16-16*(-(cr-1))))/8)
    numToSelect = (cr-1) - (8*(((numLayers-1)*(numLayers))/2))

    if fix:
        while counter < len(mcl):
            conn = []
            num = random.randint(0,len(gl)-1)
            x = gl[num].loc[0]
            y = gl[num].loc[1]
            xUpperBound = numLayers+x
            xLowerBound = x-numLayers
            yUpperBound = numLayers+y
            yLowerBound = y-numLayers

            inc = 0
            gloms = []
            act = random.uniform(0,1)
            Map.append([mcl[counter].id, num, act])
            leftover = 1-act

            # while inc < cr-1:
            #     if inc == (cr-2):
            #         act = leftover
            #     else:
            #         act = random.uniform(0, leftover)
            #         leftover -= act


            if int(numLayers) == 1:
                selected = 0
                randomGlom = generateRandomGlom(xLowerBound, xUpperBound, yLowerBound, yUpperBound, gl[0].dim[1], gl[0].dim[0])
                while selected < int(numToSelect):
                    if selected == int(numToSelect)-1:
                        act = leftover
                    else:
                        act = random.uniform(0, leftover)
                        leftover -= act
                    while randomGlom in gloms:
                        randomGlom = generateRandomGlom(xLowerBound, xUpperBound, yLowerBound, yUpperBound, gl[0].dim[1], gl[0].dim[0])
                    gloms.append(randomGlom)
                    selected += 1
                    for g in gl:
                        # if g.loc == randomGlom:
                        if g.loc == [randomGlom[0]%(gl[0].dim[1]), randomGlom[1]%(gl[0].dim[0])]:
                            num = g.id

                    Map.append([mcl[counter].id, num, act])

            elif int(numLayers) == 2:
                # get first layer first (the surrounding 8)
                xInner = x - 1
                xOuter = x + 1
                yInner = y - 1 
                yOuter = y + 1
                firstLayer = []
                while xInner <= xOuter:
                    yInner = y-1
                    while yInner <= yOuter:
                        if not((xInner == x) and (yInner == y)):
                            firstLayer.append([xInner%(gl[0].dim[1]), yInner%(gl[0].dim[0])])
                        yInner += 1
                    xInner += 1

                for a in firstLayer:
                    for g in gl:
                        if [a[0], a[1]] == g.loc:
                            num = g.id
                            act = random.uniform(0, leftover)
                            leftover -= act
                            Map.append([mcl[counter].id, num, act])

                # second layer
                selected = 0
                randomGlom = generateRandomGlom(xLowerBound, xUpperBound, yLowerBound, yUpperBound, gl[0].dim[1], gl[0].dim[0])
                while selected < int(numToSelect):
                    if selected == int(numToSelect)-1:
                        act = leftover
                    else:
                        act = random.uniform(0, leftover)
                        leftover -= act
                    while randomGlom in gloms:
                        randomGlom = generateRandomGlom(xLowerBound, xUpperBound, yLowerBound, yUpperBound, gl[0].dim[1], gl[0].dim[0])
                    gloms.append(randomGlom)
                    selected += 1
                    for g in gl:
                        # if g.loc == randomGlom:
                        if g.loc == [randomGlom[0]%(gl[0].dim[1]), randomGlom[1]%(gl[0].dim[0])]:
                            num = g.id

                    Map.append([mcl[counter].id, num, act])

            counter+=1

    return Map

def generateRandomGlom(xLowerBound, xUpperBound, yLowerBound, yUpperBound, row, col):
    """Returns a random glom location"""
    randomGlomX = random.randint(xLowerBound, xUpperBound)
    if randomGlomX == xLowerBound or randomGlomX == xUpperBound:
        randomGlomY = random.randint(yLowerBound, yUpperBound)
    else:
        randomGlomY = int(random.sample([yLowerBound, yUpperBound], 1)[0])
    # return [randomGlomX%row, randomGlomY%col]   
    return [randomGlomX, randomGlomY]    
 

def biasSample(gl, mcl, cr, fix, bias, sd=0):
    """Builds a map by choosing glomeruli to sample to mitral cells, but the more times
    a glomeruli is sampled, the less likely it is to be chosen again (either a linear
    degression or exponential based on bias). If fix != true, cr serves as mean for
    # of glom sample to each mitral cell. Weights are randomly chosen uniformly."""
    Map = []
    #determine scale
    calc = (len(mcl)/len(gl))
    scale = max(7, int(calc*1.7*cr))  
    #Build weights
    weights = _buildWeights(gl, bias, scale)
    if bias == "lin":
        s = len(weights)*scale
    else:
        s = len(weights)*(2**scale)
    
    cr_orig = cr
    counter = 0
    while counter < len(mcl):               #start looping through each mitral cell
        if not fix:
            cr = max(random.gauss(cr_orig, sd), 1)
            cr = min(cr, len(gl))
        temp = 0
        conn = []
        while temp < cr:                    #start connecting mitral cell to (multiple) glom
            rand = random.randint(1, s)
            index = 0
            while rand > 0:                 #Picking an index based on weight
                rand = rand - weights[index]
                index += 1
            index -= 1
            index = _prevDuplicates(index, conn, gl, "bias", weights, s)
            Map.append([counter, index, random.uniform(0,.4)])
            const = _recalcWeights(weights, index, bias, s)
            weights = const[0]
            s = const[1]
            conn.append(index)
            temp += 1
        counter += 1
    return Map


def _prevDuplicates(num, conn, gl, sel, weights, s):
    """If a mitral cell already connects to glom at index num, then pick
    a new number. To prevent infinite loop, if a certain number of loops
    occur, just allow duplicate but print a warning message."""
    if sel == "simple":
        check = 0
        while num in conn and check < 100:
            num = random.randint(0,len(gl)-1)
            check += 1
    else:
        check = 0
        while num in conn and check < 100:
            rand = random.randint(1, s)
            num = 0
            while rand > 0:                 #Picking an index based on weight
                rand = rand - weights[num]
                num += 1
            num -= 1
            check += 1
    if check == 100:
        print("Warning: mitral cell may be connected to same Glom cell twice in order to prevent infinite loop")
    return num
        

def _buildWeights(gl, bias, scale):
    """Returns a list len(gl), with each index starting with the same number
    which depends on bias"""
    weights = []
    counter = 0
    if bias == "lin":
        while counter < len(gl):
            weights.append(scale)
            counter += 1
    else:
        while counter < len(gl):
            weights.append(2**scale)
            counter += 1
    return weights

def _recalcWeights(weights, index, bias, s):
    """Readjusts and returns weights and sum as a 2d list [weights, sum].
    If index is too low, all inputs in weights are increased"""
    if bias == "lin":                                  
         weights[index] = weights[index] - 1
         s = s-1
    else:
        weights[index] = weights[index]/2
        s = s - weights[index]
    if weights[index] == 1:
        if bias == "lin":
            for num in range(len(weights)):
                weights[num] = weights[num] + 3
                s = s+3
        else: #bias is exp
            for num in range(len(weights)):
                x = weights[num]
                weights[num] = x*4
                s = s + x*4 - x
    return [weights, s]



###Cleaning up unconnected glom in built Map

def cleanUp(gl, mcl, Map):
    """Samples all unsampled glom in gl to the last mitral cell in mcl with a
    random uniform number for weight. **This will violate the Map if # of connections
    was fixed"""
    unsampled = []
    counter = 0
    while counter < len(gl):
        unsampled.append(counter)
        counter += 1
    counter = 0
    while counter < len(Map):
        if Map[counter][1] in unsampled:
            unsampled.remove(Map[counter][1])
        counter += 1
    while len(unsampled) > 0:
        Map.append([len(mcl)-1, unsampled[0], random.random()])
        unsampled.remove(unsampled[0])
        
def unsampledGlom(gl, mcl, Map):
    """prints out amount of gl unsampled in Map"""
    unsampled = []
    counter = 0
    while counter < len(gl):
        unsampled.append(counter)
        counter += 1
    counter = 0
    while counter < len(Map):
        if Map[counter][1] in unsampled:
            unsampled.remove(Map[counter][1])
        counter += 1
    return "Amount of unsampled Glom: " + str(len(unsampled)) + "\n"

#####Graphing
def GraphGlomActivation(gl, n, m):

    graph = [[0,0,0],[0,0,0],[0,0,0],[0,0.5,0],[0.0,1.0,0.0],[0,0.4,0],[0,0,0.4],[0,0,1],[0,0,0.8]]
    plt.imshow(graph, cmap=matplotlib.pylab.cm.YlOrRd, interpolation='nearest', origin="lower", extent=[0,3,0,3])
    plt.title("Glom Activation")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    pp = PdfPages('GlomActivation.pdf')
    pp.savefig()
    pp.close()

    plt.close()

def GraphMitralActivation(gl, mcl, n, m):
    mitralLocations = {}
    mitralActivations = {}
    for mitral in mcl:
        if str(mitral.loc) in mitralLocations:
            val = mitralLocations.get(str(mitral.loc))
            activ = mitralActivations.get(str(mitral.loc))
            mitralLocations.update({str(mitral.loc):val+1})

            activ.append(mitral.activ)
            mitralActivations.update({str(mitral.loc):activ})
        else:
            mitralLocations.update({str(mitral.loc):1})
            mitralActivations.update({str(mitral.loc):[mitral.activ]})

    maxMitrals = mitralLocations.get(max(mitralLocations, key=mitralLocations.get))

    graph = []
    counter = 0
    while counter < m*maxMitrals:
        graph.append([])
        counter1 = 0
        while counter1<m:
            graph[counter].append(0)
            counter1 += 1
        counter += 1

    for x in range(m):
        for y in range(len(graph)):
            if str([x,y/maxMitrals]) in mitralActivations:
                numActivations = len(mitralActivations.get(str([x,y/maxMitrals])))
                if numActivations == 1:
                    graph[y][x] = mitralActivations.get(str([x,y/maxMitrals]))[0]
                elif numActivations < maxMitrals:
                    if y%maxMitrals < numActivations:
                        graph[y][x] = mitralActivations.get(str([x,y/maxMitrals]))[y%maxMitrals]
                    else:
                        graph[y][x] = -0.15

                else:
                    graph[y][x] = mitralActivations.get(str([x,y/maxMitrals]))[y%(len(mitralActivations.get(str([x,y/maxMitrals]))))]


    #   https://stackoverflow.com/questions/22121239/matplotlib-imshow-default-colour-normalisation 

    fig, ax = plt.subplots()

    im = plt.imshow(graph, cmap=matplotlib.pylab.cm.YlOrRd, interpolation='nearest', vmin=-0.15, vmax=1, origin="lower", extent=[0,4,0,4])
    plt.title("Mitral Activation")
    plt.xlabel("X")
    plt.ylabel("Y")

    fig.colorbar(im)

    pp = PdfPages('MitralActivation.pdf')
    pp.savefig()
    pp.close()

    plt.close()


#####Loading and Storing a GL, MCL, and Map
def saveGL(GL, name):
    """Saves GL as a file on the computer with .GL as extention
    Precondtion: GL is a valid gl and name is a string."""
    assert type(name) == str, "name is not a string"
    st = ""
    for glom in GL:
        loc = glom.loc
        loc = str(loc[0]) + ":" + str(loc[1])
        st = st + str(glom.id) + "," + str(glom.activ) + "," + loc + "," + str(glom.conn) +";" + '\n'
    test = open(name + ".GL", "w")
    test.write(st)
    test.close

def loadGL(name):
    """Returns GL with given name from directory
    precondition: name is a string with correct extension"""
    assert type(name) == str, "name isn't a string"
    GL = []
    text = open(name)
    for line in text:
        comma1 = line.index(",")
        comma2 = line.index(",",comma1+1)
        comma3 = line.index(",",comma2+1)
        colon = line.index(":", comma2+1)
        semi = line.index(";", comma3+1)
        loc = [int(line[comma2+1:colon]), int(line[colon+1:comma3])]
        glom = cells.Glom(int(line[:comma1]), float(line[comma1+1:comma2]), loc, int(line[comma3+1:semi]))
        GL.append(glom)
    return GL

def saveMCL(MCL, name):
    """Saves MCL as a file on the computer with .MCL as extention.
    Precondition: Map is a valid map and name is a string."""
    assert type(name) == str, "name is not a string"
    st = ""
    for m in MCL:
        loc = m.loc
        loc = str(loc[0]) + ":" + str(loc[1])
        #Storing glom
        glom = str(m.glom.items())
        end = glom.index("]")
        i = 0
        s = ""
        while i < end-1:
            ind1 = glom.index("(", i)
            comma = glom.index(",",ind1)
            ind2 = glom.index(")", ind1)
            s = s + glom[ind1+1:comma] + "|" + glom[comma+2:ind2] + "+"
            i = ind2
        st = st + str(m.id) + "," + str(m.activ) + "," + loc + "," + s +";" + '\n'
    test = open(name + ".mcl", "w")
    test.write(st)
    test.close

def loadMCL(name):
    """Returns MCL with given name from directory.
    precondition: name is a string with correct extension"""
    assert type(name) == str, "name isn't a string"
    MCL = []
    text = open(name)
    for line in text:
        comma1 = line.index(",")
        comma2 = line.index(",",comma1+1)
        comma3 = line.index(",",comma2+1)
        colon = line.index(":", comma2+1)
        semi = line.index(";", comma3+1)
        loc = [int(line[comma2+1:colon]), int(line[colon+1:comma3])]
        middle = line.find("|")
        beg = comma3
        glom = {}
        while middle != -1:
            key = line[beg+1:middle]
            beg = line.index("+", beg+1)
            val = line[middle+1:beg]
            middle = line.find("|", middle+1)
            glom[int(key)] = float(val)
        mitral = cells.Mitral(int(line[:comma1]), float(line[comma1+1:comma2]), loc, glom)
        MCL.append(mitral)
    return MCL


def saveMCLSamplingMap(Map, name):
    """Saves Map as a file on the computer with .mapGLMCL as extention.
    Precondition: Map is a valid map and name is a string."""
    assert type(name) == str, "name is not a string"
    st = ""
    for elem in Map:
           st = st + str(elem[0]) + "," + str(elem[1]) + "," + str(elem[2]) + ";" + '\n'    
    test = open(name + ".mapGLMCL", "w")
    test.write(st)
    test.close

def loadMCLSamplingMap(name):
    """Returns MCL map with given name from directory. Weight value is cut off
    to the 15th decimal.
    precondition: name is a string with correct extension"""
    assert type(name) == str, "name isn't a string"
    Map = []
    text = open(name)
    for line in text:
        comma1 = line.index(",")
        comma2 = line.index(",",comma1+1)
        semi = line.index(";", comma2+1)
        Map.append([int(line[0:comma1]), int(line[comma1+1:comma2]), float(line[comma2+1:semi])])
    return Map


########Connnecting GL, MCL, and Map altogether

#How do weights play in? Right now I just do activlvl*weight
def ActivateMCLfromGL(GL, MCL, sel, Map=[], noise="None", mean=0, sd=0):
    """Builds glom connections to mitral cells and calculates mitral activ lvl based on
    connections and weights. Sel decides how to calculate the values, and noise
    adds some variation.
    **If noise = "u" then mean is the the scale for uniform distribution of 0 to mean.
    Preconditions: Map holds valid connections for GL and MCL if not empty.
    Sel = "add", "avg" or "sat". Noise = None, u, g, or e."""
    assert sel in ["add", "avg", "sat"], "select value isn't valid"
    assert noise in ["None", "u", "g", "e"], "noise isn't a valid string"
    #Build MCL - GL connections
    if Map != []:
        temp = ApplyMCLSamplingMap(GL, MCL, Map)
        MCL = temp[0]
        GL = temp[1]
    #Add noise
    if noise != "None":
        GL = addNoise(GL, noise, mean=0, sd=0)
    #Activate Mitral cell activ lvls in MCL
    if sel == "add" or sel == "avg":
        for m in MCL:
            activ = addActivationMCL(m, GL)
            if sel == "avg":
                activ = activ/(len(m.glom))
            m._activ = activ   #Bypassing assertion that activ lvl < 1
        # MCL = normalize(MCL)
    if sel == "sat":
        pass


def ApplyMCLSamplingMap(GL, MCL, Map):
    """Fills the connection details and weights for GL and MCL for the given Map.
    Returns updated MCL and GL as [MCL, GL]
    precondition: Map holds valid connections for GL and MCL"""
    assert Map[len(Map)-1][0] == len(MCL)-1, "dimensionality of Mitral cells is wrong"
    test = 0
    MCL[Map[0][0]].setLoc(GL[Map[0][1]].loc)

    for conn in Map:

        if conn[0] != test:
            test = test+1
            MCL[conn[0]].setLoc(GL[conn[1]].loc)
            MCL[conn[0]].setGlom({})
        MCL[conn[0]].glom[conn[1]]=conn[2]  #format: mc.glom[glom]=weight
        MCL[conn[0]].setGlom(MCL[conn[0]].glom)
        GL[conn[1]].setConn(GL[conn[1]].conn + 1)

    return [MCL, GL]
        
    """In List form:
    glom = MCL[conn[0]].glom
    glom.append([conn[1],conn[2]])
    MCL[conn[0]].setGlom(glom)
    GL[conn[1]].setConn(GL[conn[1]].conn + 1)"""

def addNoise(GL, noise, mean=0, sd=0):
    """Increments activation levels in GL by a certain value
    If noise is "u", then mean = scale for uniform distribution."""
    if noise == "u":
        inc = random.uniform(0,mean)
    elif noise == "g":
        inc = random.gauss(mean, sd)
    else:
        inc = random.expovariate(1/mean)
    for g in GL:
        active = max(g.activ + random.choice([1,-1])*inc, 0.0)
        g.setActiv(min(active, 1.0))
    return GL
    
def addActivationMCL(m, GL):
    """Returns updated MCL where each mitral cell's activation level is calculated
    based on adding connecting glom activation levels*weight of connection"""
    glom = m.glom.keys()
    activ = 0
    for g in glom:
        temp = GL[g].activ*m.glom[g]  #Activ lvl of attached glom * weight
        activ = activ + temp
    return activ


def normalize(MCL):
    """Given a list of Glom or PolyMitral objects the
    function will scale the highest activation value up to 1
    and the other values accordingly and return updated MCL.
    If uncomment, then firt values will scale to 0 than up to 1.
    Precondition: No activation values should be negative"""
    maxi = 0
    #mini = 100
    for m in MCL:
        assert m.activ >= 0, "Activation value was negative!"
        if maxi < m.activ:
            maxi = m.activ
        #if mini > m.activ:
        #    mini = m.activ
    #for m in MCL:
    #    m._activ = m.activ - mini   #By passing assertion that activ btwn 0 and 1
    if maxi != 0:
        scale = (1.0/maxi)  #If put mini back in then this line is 1/(maxi-mini)
        for m in MCL:
            m.setActiv(m.activ*scale)  #Assertion now in place - all #'s should be btwn 0 and 1
        return MCL
    else:
        return MCL


###### Analysis and Visualization

def euclideanDistance(layer1, layer2):
    """Returns Euclidean distance of activation levels between two layers
    of mitral or glom cells.
    Precondition: Layers are of equal length"""
    assert len(layer1) == len(layer2), "Lengths are not equal"
    index = 0
    num = 0.0
    while index < len(layer1):
        num = num + (layer1[index].activ - layer2[index].activ)**2
        index += 1
    return math.sqrt(num)

def graphLayer(layer, sort=False):
    """Returns a graph of Layer (GL or MCL) with ID # as the x axis and Activ
    level as the y axis. If sort is true, layer is sorted based on act. lvl.
    Precondition: Layer is a valid GL or MCL in order of ID with at least one element"""
    assert layer[0].id == 0, "ID's are not in order!"
    l = len(layer)
    assert l > 0, "length of layer is 0."
    if sort:
        sel_sort(layer)
    x = range(l)   #Creates a list 0...len-1
    index = 0
    y = []
    while index < l:
        y.append(layer[index].activ)
        index += 1
    plt.bar(x, y)
    if type(layer[0]) == cells.Glom:
        plt.title("Activation Levels for a Given Glomeruli Layer")
    else:
        plt.title("Activation Levels for a Given Mitral Layer")    
    plt.ylabel("Activation Level")
    plt.xlabel("Cells")
    plt.show()

def sel_sort(layer):
    """sorts layer(sel sort) from highest act lvl to lowest.
    Precondition: Layer is a valid GL or MCL"""
    i = 0
    length = len(layer)
    while i < length:
        index = _max_index(layer, i, length-1)
        _swap(layer, i, index)
        i += 1

def _max_index(layer,i,end):
    """Returns: the index of the minimum value in b[i..end]
    Precondition: b is a mutable sequence (e.g. a list)."""
    maxi = layer[i].activ
    index = i
    while i <= end:
        if layer[i].activ > maxi:
            maxi = layer[i].activ
            index = i
        i += 1
    return index


def _swap(b,x,y):
    """Procedure swaps b[h] and b[k]
    Precondition: b is a mutable sequence (e.g. a list).
    h and k are valid positions in b."""
    tmp = b[x]
    b[x] = b[y]
    b[y] = tmp

def colorMapWeights(Map, GL, MCL):
    """Builds a colormap with MCL on y axis and GL on x axis while color=weights"""
    #Create graph with all 0's
    graph = []
    for mitral in MCL:
        row = []
        for glom in GL:
            row.append(0)
        graph.append(row)
    #Add weights
    index = 0
    while index < len(Map):
        row = Map[index][0]
        col = Map[index][1]
        graph[row][col] = Map[index][2]
        index += 1
    matplotlib.pylab.matshow(graph, fignum="Research", cmap=matplotlib.pylab.cm.Greys) #Black = fully active
    plt.title("Weights in GL-MCL connection")
    plt.xlabel("GL")
    plt.ylabel("MCL")
    plt.show()
    

"""
To do list:
1. Fix Bias algorithm in biasSample() (Part of building the Map) so there's no jump in logic (need new algorithm)
2. Good saturation function for activating MCL based on connections to GL.
3. Optimize sorting algorithm to heap sort if using it often
"""
