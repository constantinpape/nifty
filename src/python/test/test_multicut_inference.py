from __future__ import print_function
import nifty
import nifty.graph
import numpy



G = nifty.graph.UndirectedGraph
CG = G.EdgeContractionGraph

GMCO = G.MulticutObjective
CGMCO = CG.MulticutObjective



def edgeContractionGraphInference(fmSubFac):

    chainLength = 10
    chainEdges =[]
    for x in range(chainLength-1):
        chainEdges.append((x,x+1))


    class Callback(nifty.graph.EdgeContractionGraphCallback):
        def __init__(self):
            super(Callback, self).__init__()
        def contractEdge(self, edge):
            pass
        def mergeEdges(self, alive, dead):
            pass
        def mergeNodes(self, alive, dead):
            pass
        def contractEdgeDone(self, edge):
            pass



    g =  G(chainLength)
    g.insertEdges(chainEdges)

    callback = Callback()
    cg = CG(g, callback)

    assert cg.numberOfEdges == chainLength - 1
    assert cg.numberOfNodes == chainLength



    while cg.numberOfEdges != 0:
        lastEdge = cg.numberOfEdges - 1
        cg.contractEdge(lastEdge)
        weights = numpy.zeros(chainLength-1)
        obj = nifty.graph.multicut.multicutObjective(cg, weights)



        fusionMove = CGMCO.fusionMoveSettings(mcFactory=fmSubFac)
        fmFac = CGMCO.fusionMoveBasedFactory(fusionMove=fusionMove,
            numberOfThreads=-1,
            proposalGen=CGMCO.watershedProposals()
            #numberOfIterations=200,
            #stopIfNoImprovement=200
        )
        solver = fmFac.create(obj)
        # run inference
        #visitor = obj.multicutVerboseVisitor()
        ret = solver.optimize()#visitor=visitor)
       

def testEdgeContractionGraphInferenceFmGreedy():
    fmSubFac = GMCO.greedyAdditiveFactory()
    edgeContractionGraphInference(fmSubFac=None)


if nifty.Configuration.WITH_CPLEX:

   def testEdgeContractionGraphInferenceFmCplex():
       fmSubFac = GMCO.multicutIlpCplexFactory()
       edgeContractionGraphInference(fmSubFac=fmSubFac)

if nifty.Configuration.WITH_GUROBI:

   def testEdgeContractionGraphInferenceFmGurobi():
       fmSubFac = GMCO.multicutIlpGurobiFactory()
       edgeContractionGraphInference(fmSubFac=fmSubFac)

if nifty.Configuration.WITH_GLPK:

   def testEdgeContractionGraphInferenceFmGlpk():
       fmSubFac = GMCO.multicutIlpGlpkFactory()
       edgeContractionGraphInference(fmSubFac=fmSubFac)


def testEdgeContractionGraphInferenceGreedyAdditive():

    chainLength = 10
    chainEdges =[]
    for x in range(chainLength-1):
        chainEdges.append((x,x+1))


    class Callback(nifty.graph.EdgeContractionGraphCallback):
        def __init__(self):
            super(Callback, self).__init__()
        def contractEdge(self, edge):
            pass
        def mergeEdges(self, alive, dead):
            pass
        def mergeNodes(self, alive, dead):
            pass
        def contractEdgeDone(self, edge):
            pass



    g =  G(chainLength)
    g.insertEdges(chainEdges)
    callback = Callback()
    cg = CG(g, callback)

    assert cg.numberOfEdges == chainLength - 1
    assert cg.numberOfNodes == chainLength



    while cg.numberOfEdges != 0:

        lastEdge = cg.numberOfEdges - 1
        cg.contractEdge(lastEdge)
        weights = numpy.zeros(chainLength-1)
        obj = nifty.graph.multicut.multicutObjective(cg, weights)



        factory = CGMCO.greedyAdditiveFactory()
        solver = factory.create(obj)
        #visitor = obj.multicutVerboseVisitor()
        ret = solver.optimize()#visitor=visitor)


