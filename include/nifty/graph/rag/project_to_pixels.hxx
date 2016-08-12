#pragma once
#ifndef NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX
#define NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX


#include <algorithm>

#include "nifty/marray/marray.hxx"
#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/array/arithmetic_array.hxx"

//#include "nifty/graph/detail/contiguous_indices.hxx"

#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif


namespace nifty{
namespace graph{



template<
    size_t DIM, 
    class LABELS_TYPE, 
    class PIXEL_ARRAY, 
    class NODE_MAP
>
void projectScalarNodeDataToPixels(
    const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
    NODE_MAP & nodeData,
    PIXEL_ARRAY & pixelData,
    const int numberOfThreads = -1
){
    typedef array::StaticArray<int64_t, DIM> Coord;

    const auto labelsProxy = graph.labelsProxy();
    const auto & shape = labelsProxy.shape();
    const auto labels = labelsProxy.labels(); 
    

    // if scalar 
    auto pOpt = nifty::parallel::ParallelOptions(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpt);
    nifty::tools::parallelForEachCoordinate(threadpool, shape,
    [&](int tid, const Coord & coord){
        const auto node = labels(coord.asStdArray());
        pixelData(coord.asStdArray()) = nodeData[node];
    });

}

template<
    class LABELS_PROXY, 
    class PIXEL_ARRAY, 
    class NODE_MAP
>
void projectScalarNodeDataToPixels(
    const GridRagStacked2D<LABELS_PROXY> & graph,
    NODE_MAP & nodeData, // why is this not const?
    PIXEL_ARRAY & pixelData,
    const int numberOfThreads = -1
){
    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;

    const auto & labelsProxy = graph.labelsProxy();
    const auto & shape = labelsProxy.shape();
        
    typedef LABELS_PROXY LabelsProxyType;
    typedef typename LABELS_PROXY::LabelType LabelType;
    typedef typename LabelsProxyType::BlockStorageType LabelsBlockStorage;
    typedef typename tools::BlockStorageSelector<PIXEL_ARRAY>::type DataBlockStorage;
    
    nifty::parallel::ParallelOptions pOpts(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpts);
    const auto nThreads = pOpts.getActualNumThreads();
    
    uint64_t numberOfSlices = shape[0];
    Coord2 sliceShape2({shape[1], shape[2]});
    Coord sliceShape3({int64_t(1),shape[1], shape[2]});
        
    LabelsBlockStorage sliceLabelsStorage(threadpool, sliceShape3, nThreads);
    DataBlockStorage   sliceDataStorage(threadpool, sliceShape3, nThreads);

    parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){

        // fetch the data for the slice
        auto sliceLabelsFlat3DView = sliceLabelsStorage.getView(tid);
        auto sliceDataFlat3DView   = sliceDataStorage.getView(tid);
        
        const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
        const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
        
        tools::readSubarray(labelsProxy, blockBegin, blockEnd, sliceLabelsFlat3DView);
        // TODO do we have to initialize slicedData ?
        
        auto sliceLabels = sliceLabelsFlat3DView.squeezedView();
        auto sliceData = sliceDataFlat3DView.squeezedView();
        
        nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
            const auto node = sliceLabels( coord.asStdArray() );            
            sliceData(coord.asStdArray()) = nodeData[node];
        });
        
        tools::writeSubarray(pixelData, blockBegin, blockEnd, sliceDataFlat3DView);

    });

}


} // namespace graph
} // namespace nifty



#endif /* NIFTY_GRAPH_RAG_PROJECT_TO_PIXELS_HXX */
