#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX


#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/marray/marray.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#endif

#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace graph{


    template<size_t DIM, class LABELS_TYPE, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const ExplicitLabelsGridRag<DIM, LABELS_TYPE> & graph,
        nifty::marray::View<LABELS> data,
        NODE_MAP &  nodeMap
    ){
        typedef std::array<int64_t, DIM> Coord;

        const auto labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const auto labels = labelsProxy.labels(); 

        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());
        


        nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
            const auto node = labels(coord);
            const auto l  = data(coord);
            overlaps[node][l] += 1;
        });

        for(const auto node : graph.nodes()){
            const auto & ol = overlaps[node];
            // find max ol 
            uint64_t maxOl = 0 ;
            uint64_t maxOlLabel = 0;
            for(auto kv : ol){
                if(kv.second > maxOl){
                    maxOl = kv.second;
                    maxOlLabel = kv.first;
                }
            }
            nodeMap[node] = maxOlLabel;
        }
    }
    
    template<class LABELS_PROXY, class LABELS, class NODE_MAP>
    void gridRagAccumulateLabels(
        const GridRagStacked2D<LABELS_PROXY> & graph,
        const LABELS & data,                         
        NODE_MAP &  nodeMap,
        const int numberOfThreads = -1
    ){
        
        typedef LABELS_PROXY LabelsProxyType;
        typedef typename LABELS_PROXY::LabelType LabelType;
        typedef typename LabelsProxyType::BlockStorageType LabelsBlockStorage;
        typedef typename tools::BlockStorageSelector<LABELS>::type DataBlockStorage;
        
        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = graph.labelsProxy();
        const auto & shape = labelsProxy.shape();
        
        NIFTY_CHECK_OP(data.shape(0),==,shape[0], "Shape along z does not agree")
        NIFTY_CHECK_OP(data.shape(1),==,shape[1], "Shape along y does not agree")
        NIFTY_CHECK_OP(data.shape(2),==,shape[2], "Shape along x does not agree")
        
        nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();
        
        uint64_t numberOfSlices = shape[0];
        Coord2 sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});

        std::vector<  std::unordered_map<uint64_t, uint64_t> > overlaps(graph.numberOfNodes());
        
        LabelsBlockStorage sliceLabelsStorage(threadpool, sliceShape3, nThreads);
        DataBlockStorage   sliceDataStorage(threadpool, sliceShape3, nThreads);

        parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){

            // fetch the data for the slice
            auto sliceLabelsFlat3DView = sliceLabelsStorage.getView(tid);
            auto sliceDataFlat3DView   = sliceDataStorage.getView(tid);
            
            const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
            const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
            
            tools::readSubarray(labelsProxy, blockBegin, blockEnd, sliceLabelsFlat3DView);
            tools::readSubarray(data, blockBegin, blockEnd, sliceDataFlat3DView);
            
            auto sliceLabels = sliceLabelsFlat3DView.squeezedView();
            auto sliceData = sliceDataFlat3DView.squeezedView();
            
            nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                const auto node = sliceLabels( coord.asStdArray() );            
                const auto l    = sliceData( coord.asStdArray() );
                overlaps[node][l] += 1;
            });

        });
        
        parallel::parallel_foreach(threadpool, graph.numberOfNodes(), [&](const int tid, const int64_t nodeId){
            const auto & ol = overlaps[nodeId];
            // find max ol 
            uint64_t maxOl = 0 ;
            uint64_t maxOlLabel = 0;
            for(auto kv : ol){
                if(kv.second > maxOl){
                    maxOl = kv.second;
                    maxOlLabel = kv.first;
                }
            }
            nodeMap[nodeId] = maxOlLabel;
        });
    }


} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_GRID_RAG_FEATURES_HXX */
