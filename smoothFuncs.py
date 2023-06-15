# Mitchell Gronowitz
# Running dPsiSaturation with separate excel docs and graph functions

# Edited by Christopher De Jesus
# Summer 2023

from __future__ import annotations

from RnO import (
    QSpace, Odorscene, Ligand, createEpithelium, saveEpithelium, loadEpithelium,
    dPsiBarSaturation, colorMapSumOfSquares, dPsiGraphFromExcel, graphFromExcel,
    dPsiOccActGraphFromExcel, glom_penetrance, peak_affinity
)
import time
import multiprocessing

def testdPsiBarSat(fixed, aff_sd=None, eff_sd=None, numRecs=30, c=1, dim=2, qspaces=None, purpose="standard"):
    """Runs multiple graphs of given qspaces at one time
    Optional - run makeSimilar (line 25), to create 3 epitheliums with equal eff and aff SD's (only rec means differ)
    Otherwise - make sure there are three saved epithelium files with correct names

    Returns a dPsiBar graph with three qspaces, an act and occ graph, and excel docs with details
    
    fixed = True if want eff = 1
    c = convergence ratio of recs to glom
    purpose = reason for running simulation = either "eff", "aff", "c", "recs", "redAff", "dim", or 'standard'"""
    
    #Run this function if don't already have saved epithelium files to use

    if aff_sd is None:
        aff_sd = [0.5, 1.5]
    if eff_sd is None:
        eff_sd = [0.05, 1.0]
    if qspaces is None:
        qspaces = [4, 10, 30]

    makeSimilar(numRecs, aff_sd, eff_sd, purpose, qspaces, dim)
    
    startTime = time.time()
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, c, dim)
    
    pdfName = "LigandSat with varying qspaces" + purp
    plotTitle = "Saturation of dPsiBar" + purp
    
    #i = 0
    labelNames = []
    excelNames = []
    end = False
    
    # FIXME: why
    if __name__ == '__main__':
        jobs = []
    
    
        #while i < len(qspaces):
        for i, qspacesItem in enumerate(qspaces):    
            space = []
            #j = 0
            #while j < dim:
            for j in range(dim):    
                space.append((0,qspacesItem))
                #j+=1
            qspace = QSpace(space)
            epith = loadEpithelium("1. SavedEpi_" + str(qspace.size[0]) + purp + ".csv")

            labelNames.append(str(qspace.size[0]) + " qspace")
            excelNames.append("LigandSat with " + str(qspace.size[0]) + " qspace" + purp)
        
            if i == (len(qspaces) - 1):
                end = True
          
            p = multiprocessing.Process(target=dPsiBarSaturation, args=(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, False))
            jobs.append(p)
            p.start()
        
        
            #epi, dn, qspace, pdfName, labelName, excelName, fixed eff
            #dPsiBarSaturation(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, graphIt=False)
        
            #i += 1
            print("Graph #" + str(i) + ": " + str((time.time() - startTime) / 60.0 ) + " minutes")

            print("Overall time: " + str((time.time() - startTime) / 60.0 ) + " minutes")
        
        
        for j, job in enumerate(jobs):
            job.join()
        

# FIXME: Lists in parameter space
def testdPsiBarSatColorMap(fixed, aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4,10,30], purpose="standard", qunits = 3):
    """Runs multiple graphs of given qspaces at one time
    Optional - run makeSimilar (line 25), to create 3 epitheliums with equal eff and aff SD's (only rec means differ)
    Otherwise - make sure there are three saved epithelium files with correct names

    Returns a dPsiBar graph with three qspaces, an act and occ graph, and excel docs with details
    
    fixed = True if want eff = 1
    c = convergence ratio of recs to glom
    purpose = reason for running simulation = either "eff", "aff", "c", "recs", "redAff", "dim", or 'standard'"""
    
    #Run this function if don't already have saved epithelium files to use
    makeSimilar(numRecs, aff_sd, eff_sd, purpose, qspaces, dim)
    
    startTime = time.time()
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, c, dim)
    
    pdfName = "LigandSat with varying qspaces" + purp
    plotTitle = "Saturation of dPsiBar" + purp
    
    #i = 0
    labelNames = []
    excelNames = []
    end = False
    
    
    # if __name__ == '__main__':
    #     jobs = []
    print("BEFORE FOR LOOP!!!!!!")
    
    #while i < len(qspaces):
    for i, qspacesItem in enumerate(qspaces):    
        space = []
        #j = 0
        #while j < dim:
        for j in range(dim):    
            space.append((0,qspacesItem))
            #j+=1
        qspace = QSpace(space)

        print("QSPACE SIZE IS " + str(qspace.size[0]) + "????????")

        epith = loadEpithelium("1. SavedEpi_" + str(qspace.size[0]) + purp + ".csv")

        print("LOADED EPITHELIUM")
        labelNames.append(str(qspace.size[0]) + " qspace")
        excelNames.append("LigandSat with " + str(qspace.size[0]) + " qspace" + purp)
    
        # if i == (len(qspaces) - 1):
        #     end = True

        x = 0
        y = 0
        ID = 0
        odorscenes = []
        #while x < qspace.size[0][1]:
        while x < qunits*10:
            y = 0
            #while y < qspace.size[1][1]:
            while y < qunits*10:
                odorscenes.append(Odorscene(x,[Ligand(ID, [x/float(qunits),y/float(qunits)], .004)]))
                y += 1
                ID += 1
            x += 1
        print("POPULATED ODORSCENES")
        colorMapSumOfSquares(epith, odorscenes, .3, qspace)
      
        # p = multiprocessing.Process(target=dPsiBarSaturation, args=(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, False))
        # jobs.append(p)
        # p.start()
    
    
        #epi, dn, qspace, pdfName, labelName, excelName, fixed eff
        #dPsiBarSaturation(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, graphIt=False)
    
        #i += 1
        print("Graph #" + str(i) + ": " + str((time.time() - startTime) / 60.0 ) + " minutes")

        print("Overall time: " + str((time.time() - startTime) / 60.0 ) + " minutes")
        
        
        # for j, job in enumerate(jobs):
        #     job.join()


def testdPsiBarSaturationDim(dims, fixed=False, aff_sd=[.5,1.5], eff_sd=[.05,1.0], numRecs=30, c=1):
    """Runs 4 simulations of differing dimensions determined by dims all with (0,4) qspace.
    Since each simulation has an added dimension, it wasn't possible
    to make the epithelium identical. Therefore, all means, aff and eff
    are randomized between a constant distribution.
    Returns a dPsiBar graph with dimensions specified in dims. Also returns act and occ graphs
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
        saveEpithelium(epith, "1. SavedEpi_(0, 4), dim=" + str(dim))
        
        labels.append(str(dim) + "D")
        excels.append("LigandSat with (0, 4) qspace, dim=" + str(dim))
        
        if index == (len(dims) - 1):
            end = True

        dPsiBarSaturation(epith, .01, qspace, pdfName, labels[index], excels[index], fixed, c, plotTitle, end,  ', dim=' + str(dim), False)
        index += 1
        
    print("Overall time: " + str((time.time() - startTime) / 60.0 ) + " minutes")
    

def allGraphsFromExcel(aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4,10,30], purpose="standard", rep=200.0):
    """Given excel docs in correct directories, this creates a dpsiSaturation graph and act and occ graphs"""
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, c, dim)
    xaxis = [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,200,250,300,350,400] #xaxis for dPsi vs num of ligands
    
    ####Creating dPsiSaturation graphs
    i = 0
    pdfName = "LigandSat with varying qspaces" + purp
    titleName = "Saturation of dPsiBar" + purp
    qspaceList: list[QSpace]=[]
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
    saveEpithelium(epith, "1. SavedEpi_" + str(qspace.size[0]) + purp)
    
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
        for rec in epith2.recs:
            rec.sdA = epith.recs[k].sdA
            rec.sdE = epith.recs[k].sdE
            rec.covA = None
            rec.covE = None       
            k += 1
    
        saveEpithelium(epith2, "1. SavedEpi_" + str(qspace.size[0]) + purp)
        
        i += 1

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
    print("start of smoothFuncs:"+str(time.time()))

    # runReceptorOdorGraphToolStandAlone()

    testdPsiBarSat(fixed=False, aff_sd=[.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4,10,30], purpose="standard")

    # testdPsiBarSatColorMap(fixed=True, aff_sd=[.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4], purpose="standard", qunits = 4)


    # allGraphsFromExcel(aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4,10,30], purpose="standard", rep=2)

    ####testing varying dimensions
    #testdPsiBarSaturationDim(dims=[2,3,4,5], fixed=False, aff_sd=[.5,1.5], eff_sd=[.05,1.0], numRecs=30, c=1)
    #dimAllGraphsFromExcel(numRecs=30, dims=[2,3,4,5], rep=200)

    print("end of smoothFuncs:"+str(time.time()))
    
test()
