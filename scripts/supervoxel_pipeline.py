from __future__ import print_function
from __future__ import division

import nifty
import nifty.viewer
import numpy
import vigra
import h5py 

import nifty.tools
from multiprocessing import cpu_count
import pylab
import scipy.ndimage
import math
import threading
import fastfilters

pmapPath = "/home/tbeier/prediction_semantic_binary_full.h5"

heightMapFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/heightMap2.h5"
oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/ufd_overseg3.h5"
oversegFile = "/home/tbeier/ufd_overseg3.h5"
#oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/aggloseg_0.3_50.h5"
agglosegBaseFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/aggloseg_"




class DummyWithStatement:
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass



def makeBall(r):
    size = 2*r + 1

    mask = numpy.zeros([size]*3)

    for x0 in range(-1*r, r + 1):
        for x1 in range(-1*r, r + 1):
            for x2 in range(-1*r, r + 1):
                
                if math.sqrt(x0**2 + x1**2 + x2**2) <= r:
                    mask[x0+r, x1+r, x2+r] = 1

    return mask, (r,r,r)


def membraneOverseg3D(pmapDset, heightMapDset, **kwargs):



    axisResolution = kwargs.get("axisResolution",['4nm']*3)
    featureBlockShape = kwargs.get("featureBlockShape",['100']*3)
    shape = pmapDset.shape[0:3]
    

    roiBegin = kwargs.get("roiBegin", [0]*3)
    roiEnd = kwargs.get("roiEnd", shape)
    nWorkers = kwargs.get("nWorkers",cpu_count())
    invertPmap = kwargs.get("invertPmap",False)

    blocking = nifty.tools.blocking(roiBegin=roiBegin, roiEnd=roiEnd, blockShape=featureBlockShape)
    margin = [45 ,45,45]


    def pmapToHeightMap(pmap):
        
        footprint, origin = makeBall(r=3)

        medianImg = scipy.ndimage.percentile_filter(input=pmap, 
                                                    #size=(20,20,20),
                                                    footprint=footprint, 
                                                    #origin=origin, 
                                                    mode='reflect',
                                                    percentile=50.0)
        if False:
            blurredSmall = vigra.gaussianSmoothing(pmap.T, 1.0,).T
            blurredLarge = vigra.gaussianSmoothing(pmap.T, 6.0,).T
            blurredSuperLarge = vigra.gaussianSmoothing(pmap.T, 10.0,).T

        else:
            blurredSmall = fastfilters.gaussianSmoothing(pmap, 1.0,)
            blurredLarge = fastfilters.gaussianSmoothing(pmap, 6.0,)
            blurredSuperLarge = fastfilters.gaussianSmoothing(pmap, 10.0,)

        combined = medianImg + blurredSuperLarge*0.3 + 0.15*blurredLarge + 0.1*blurredSmall

        footprint, origin = makeBall(r=3)
        combined = scipy.ndimage.percentile_filter(input=combined, 
                                                    #size=(20,20,20),
                                                    footprint=footprint, 
                                                    #origin=origin, 
                                                    mode='reflect',
                                                    percentile=50.0)

        combined = fastfilters.gaussianSmoothing(combined, 1.3)


        if False:
            nifty.viewer.view3D(pmap, show=False, title='pm',cmap='gray')
            nifty.viewer.view3D(medianImg, show=False, title='medianImg',cmap='gray')
            nifty.viewer.view3D(combined, show=False, title='combined',cmap='gray')
            pylab.show()

        return combined


    numberOfBlocks = blocking.numberOfBlocks
    lock = threading.Lock()
    noLock = DummyWithStatement()
    done = [0]


    for blockIndex in range(numberOfBlocks):
        blockWithHalo = blocking.getBlockWithHalo(blockIndex, margin)
        block = blocking.getBlock(blockIndex)
        outerBlock = blockWithHalo.outerBlock
        innerBlock = blockWithHalo.innerBlock
        innerBlockLocal = blockWithHalo.innerBlockLocal

        #print("B ",block.begin, block.end)
        #print("O ",outerBlock.begin, outerBlock.end)
        #print("I ",innerBlock.begin, innerBlock.end)
        #print("IL",innerBlockLocal.begin, innerBlockLocal.end)



    with nifty.tools.progressBar(size=numberOfBlocks) as bar:

        def f(blockIndex):
            blockWithHalo = blocking.getBlockWithHalo(blockIndex, margin, margin)
            #print "fo"
            outerBlock = blockWithHalo.outerBlock
            outerSlicing = nifty.tools.getSlicing(outerBlock.begin, outerBlock.end)
            b,e = outerBlock.begin, outerBlock.end
            #print bi
            #
            
            maybeLock = [lock, noLock][isinstance(pmapDset, numpy.ndarray)]
            
            with maybeLock:
                if invertPmap:
                    outerPmap = 1.0 - pmapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]
                else:
                    outerPmap = pmapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]

            heightMap = pmapToHeightMap(outerPmap)

            # 
            innerBlockLocal = blockWithHalo.innerBlockLocal
            b,e = innerBlockLocal.begin, innerBlockLocal.end
            innerHeightMap = heightMap[b[0]:e[0], b[1]:e[1], b[2]:e[2]]


            b,e =  blockWithHalo.innerBlock.begin,  blockWithHalo.innerBlock.end

            if isinstance(heightMapDset,numpy.ndarray):
                print("NOT LOCKED")
                heightMapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = innerHeightMap
                with lock:
                    done[0] += 1
                    bar.update(done[0])

            else:
                with lock:
                    print("locked",b,e)

                    heightMapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = innerHeightMap
                    done[0] += 1
                    bar.update(done[0])


        nifty.tools.parallelForEach(range(blocking.numberOfBlocks), f=f, nWorkers=nWorkers)






def makeSmallerSegNifty(oseg,  volume_feat, reduceBySetttings, wardnessSettings, baseFilename):
    import nifty
    import nifty.graph
    import nifty.graph.rag
    import nifty.graph.agglo

    nrag = nifty.graph.rag
    nagglo = nifty.graph.agglo

    print("overseg in c order starting at zero")
    oseg = numpy.require(oseg, dtype='uint32',requirements='C')
    oseg -= 1

    print("make rag")
    rag = nifty.graph.rag.gridRag(oseg)

    print("volfeatshape")
    vFeat = numpy.require(volume_feat, dtype='float32',requirements='C')

    print("accumulate means and counts")
    eFeatures, nFeatures = nrag.accumulateMeanAndLength(rag, vFeat, [100,100,100],-1)

    eMeans = eFeatures[:,0]
    eSizes = eFeatures[:,1]
    nSizes = nFeatures[:,1]

    print("get clusterPolicy")


    for wardness in wardnessSettings:
        for reduceBy in reduceBySetttings:

            print("wardness",wardness,"reduceBy",reduceBy)

            numberOfNodesStop = int(float(rag.numberOfNodes)/float(reduceBy) + 0.5)
            numberOfNodesStop = max(1,numberOfNodesStop)
            numberOfNodesStop = min(rag.numberOfNodes, numberOfNodesStop)


            clusterPolicy = nagglo.edgeWeightedClusterPolicy(
                graph=rag, edgeIndicators=eMeans,
                edgeSizes=eSizes, nodeSizes=nSizes,
                numberOfNodesStop=numberOfNodesStop,
                sizeRegularizer=float(wardness))

            print("do clustering")
            agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy) 
            agglomerativeClustering.run()
            seg = agglomerativeClustering.result()#out=[1,2,3,4])

            print("make seg dense")
            dseg = nifty.tools.makeDense(seg)

            print(dseg.dtype, type(dseg))

            print("project to pixels")
            pixelData = nrag.projectScalarNodeDataToPixels(rag, dseg.astype('uint32'))
            print("done")
            
            outFilename = baseFilename + str(wardness) + "_" + str(reduceBy) + ".h5"

            agglosegH5 = h5py.File(outFilename,'w')
            agglosegDset = agglosegH5.create_dataset('data',shape=pmapDset.shape[0:3], chunks=(100,100,100),dtype='uint32',compression="gzip")
            agglosegDset[:,:,:] = pixelData
            agglosegH5.close()



if True:
    pmapH5 = h5py.File(pmapPath,'r')
    pmapDset = pmapH5['data']
    shape = list(pmapDset.shape[0:3])

    subset = None
    sshape = (subset,)*3
    heightMapH5 = h5py.File(heightMapFile,'w')
    heightMapDset = heightMapH5.create_dataset('data',shape=shape,dtype='float32')



    print("load pmap in ram (since we have enough")
    if subset is None:
        pmapArray= pmapDset[:,:,:,:]
    else:
        pmapArray= pmapDset[0:subset,0:subset,0:subset,:]
    pmapArray = pmapArray[:,:,:,0]
    pmapH5.close()



    with vigra.Timer("st"):
        params = {
            "axisResolution" :  [2.0, 2.0, 2.0],
            "featureBlockShape" : [350,350,350],
            "invertPmap": False,
            #"roiBegin": [0,0,0],
            #"roiEnd":   sshape
            #"nWorkers":1,
        }
        membraneOverseg3D(pmapArray,heightMapDset, **params)


    heightMapH5.close()


if True:

    print("read hmap")
    heightMapH5 = h5py.File(heightMapFile,'r')
    heightMapDset = heightMapH5['data']
    heightMap = heightMapDset[:,:,:]
    shape = list(heightMap.shape)
    heightMapH5.close()

    print("do overseg")
    overseg, nseg = vigra.analysis.unionFindWatershed3D(heightMap.T, blockShape=(100,100,100))
    overseg = overseg.T
    oversegH5 = h5py.File(oversegFile)

    overseg = numpy.array(overseg)

    print("write")
    oversegDset = oversegH5.create_dataset('data',data=overseg, chunks=(100,100,100))#,compression='gzip')
    oversegDset.attrs['nseg'] = nseg
    oversegH5.close()
   

    





 

if False:

    import nifty.hdf5

    with vigra.Timer("ws"):
        nifty.hdf5.unionFindWatershed(heightMapFile,'data', oversegFile,'data',[100,100,100])
    


if True:

    oversegH5 = h5py.File(oversegFile,'r')
    oversegDset = oversegH5['data']

    blocking = nifty.tools.blocking(roiBegin=(0,0,0), roiEnd=oversegDset.shape[0:3], blockShape=[128,128,128])


    numberOfBlocks = blocking.numberOfBlocks
    lock = threading.Lock()
    done = [0]

    with nifty.tools.progressBar(size=numberOfBlocks) as bar:

        def f(blockIndex):
            block = blocking.getBlock(blockIndex)
            b,e = block.begin, block.end
          
            labels = oversegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]
            oversegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = labels - 1


            with lock:
                heightMapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = innerHeightMap
                done[0] += 1
                bar.update(done[0])
        nifty.tools.parallelForEach(range(blocking.numberOfBlocks), f=f, nWorkers=nWorkers)




if True:


    heightMapH5 = h5py.File(heightMapFile,'r')
    heightMapDset = heightMapH5['data']

    oversegH5 = h5py.File(oversegFile,'r')
    oversegDset = oversegH5['data']




    print("read hmap")
    heightMap = heightMapDset[:,:,:]

    print("read oseg")
    overseg = oversegDset[:,:,:]

    print("make smaller")
    makeSmallerSegNifty(overseg,heightMap, [5,10,20], [0.3], agglosegBaseFile)



    oversegH5.close()
    pmapH5.close()
    #agglosegH5.close()