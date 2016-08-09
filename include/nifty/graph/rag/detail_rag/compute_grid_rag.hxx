#pragma once
#ifndef NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HXX
#define NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HXX

#include <functional>
#include <algorithm>

#include "nifty/container/boost_flat_set.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/for_each_coordinate.hxx"


namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class ExplicitLabels;

template<class LABEL_TYPE>
class GridRagStacked2D;

template<size_t DIM, class LABELS_PROXY>
class GridRag;



namespace detail_rag{

template< class GRID_RAG>
struct ComputeRag;


template<size_t DIM, class LABEL_TYPE>
struct ComputeRag< GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> > > {

    template<class S>
    static void computeRag(
        GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> > & rag,
        const S & settings
    ){
        typedef GridRag<DIM,  ExplicitLabels<DIM, LABEL_TYPE> >  Graph;
        typedef array::StaticArray<int64_t, DIM> Coord;
        typedef typename Graph::NodeAdjacency NodeAdjacency;
        typedef typename Graph::EdgeStorage EdgeStorage;

        const auto labelsProxy = rag.labelsProxy();
        const auto numberOfLabels = labelsProxy.numberOfLabels();
        const auto labels = labelsProxy.labels(); 
        const auto & shape = labelsProxy.shape();

        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);

        // assign the number of nodes to the graph
        rag.assign(numberOfLabels);

        auto makeCoord2 = [](const Coord & coord,const size_t axis){
            Coord coord2 = coord;
            coord2[axis] += 1;
            return coord2;
        };

        if(pOpts.getActualNumThreads()<=1){
            nifty::tools::forEachCoordinate(shape,[&](const Coord & coord){
                const auto lU = labels(coord.asStdArray());
                for(size_t axis=0; axis<DIM; ++axis){
                    auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < shape[axis]){
                        const auto lV = labels(coord2.asStdArray());
                        if(lU != lV){
                            rag.insertEdge(lU,lV);
                        }
                    }
                }
            });
        }
        else{
            nifty::parallel::ThreadPool threadpool(pOpts);
            struct PerThread{
                std::vector< container::BoostFlatSet<uint64_t> > adjacency;
            };

            std::vector<PerThread> perThreadDataVec(pOpts.getActualNumThreads());
            for(size_t i=0; i<perThreadDataVec.size(); ++i)
                perThreadDataVec[i].adjacency.resize(numberOfLabels);

            // collect the node-adjacency sets in parallel which needs to be merged later 
            nifty::tools::parallelForEachCoordinate(threadpool, shape,[&](const int tid, const Coord & coord){
                auto & adjacency = perThreadDataVec[tid].adjacency;
                const auto lU = labels(coord.asStdArray());
                for(size_t axis=0; axis<DIM; ++axis){
                    const auto coord2 = makeCoord2(coord, axis);
                    if(coord2[axis] < shape[axis]){
                        const auto lV = labels(coord2.asStdArray());
                        if(lU != lV){
                            adjacency[lV].insert(lU);
                            adjacency[lU].insert(lV);
                        }
                    }
                }
            });

            rag.mergeAdjacencies(perThreadDataVec, threadpool);
        }
    }
};



template<class LABELS_PROXY>
struct ComputeRag< GridRagStacked2D< LABELS_PROXY > > {

    typedef LABELS_PROXY LabelsProxyType;
    typedef typename LabelsProxyType::BlockStorageType BlockStorageType;
    typedef GridRagStacked2D< LABELS_PROXY > RagType;
    typedef typename LABELS_PROXY::LabelType LabelType;
    
    template<class S>
    static void computeRag(
        RagType & rag,
        const S & settings
    ){
        std::cout<<"\nphase 0\n";

        typedef array::StaticArray<int64_t, 3> Coord;
        typedef array::StaticArray<int64_t, 2> Coord2;

        const auto & labelsProxy = rag.labelsProxy();
        const auto & shape = labelsProxy.shape();
        const LabelType numberOfLabels = labelsProxy.numberOfLabels();
        
        rag.assign(numberOfLabels);

        nifty::parallel::ParallelOptions pOpts(settings.numberOfThreads);
        nifty::parallel::ThreadPool threadpool(pOpts);
        const auto nThreads = pOpts.getActualNumThreads();

        uint64_t numberOfSlices = shape[0];
        array::StaticArray<int64_t, 2> sliceShape2({shape[1], shape[2]});
        Coord sliceShape3({int64_t(1),shape[1], shape[2]});

        auto & perSliceDataVec = rag.perSliceDataVec_;

        /////////////////////////////////////////////////////
        // Phase 1 : In slice node adjacency and edge count
        /////////////////////////////////////////////////////
        std::cout<<"phase 1\n";
        { 


            BlockStorageType sliceLabelsStorage(threadpool, sliceShape3, nThreads);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                // 
                auto sliceLabelsFlat3DView = sliceLabelsStorage.getView(tid);

                // fetch the data for the slice
                const Coord blockBegin({sliceIndex,int64_t(0),int64_t(0)});
                const Coord blockEnd({sliceIndex+1, sliceShape2[0], sliceShape2[1]});
                labelsProxy.readSubarray(blockBegin, blockEnd, sliceLabelsFlat3DView);
                auto sliceLabels = sliceLabelsFlat3DView.squeezedView();

                // do the thing 
                nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                    const auto lU = sliceLabels(coord.asStdArray());
                    sliceData.minInSliceNode = std::min(sliceData.minInSliceNode, lU);
                    sliceData.maxInSliceNode = std::max(sliceData.maxInSliceNode, lU);
                    for(size_t axis=0; axis<2; ++axis){
                        Coord2 coord2 = coord;
                        ++coord2[axis];
                        if(coord2[axis] < sliceShape2[axis]){
                            const auto lV = sliceLabels(coord2.asStdArray());
                            if(lU != lV){
                                sliceData.minInSliceNode = std::min(sliceData.minInSliceNode, lV);
                                sliceData.maxInSliceNode = std::max(sliceData.maxInSliceNode, lV);
                                if(rag.insertEdgeOnlyInNodeAdj(lU, lV)){
                                    ++perSliceDataVec[sliceIndex].numberOfInSliceEdges;
                                }
                            }
                        }
                    }
                });
            });
        }

        std::cout<<"phase 2\n";
        /////////////////////////////////////////////////////
        // Phase 2 : set up the in slice edge offsets
        /////////////////////////////////////////////////////
        {
            for(auto sliceIndex=1; sliceIndex<numberOfSlices; ++sliceIndex){
                const auto prevOffset = perSliceDataVec[sliceIndex-1].inSliceEdgeOffset;
                const auto prevEdgeNum = perSliceDataVec[sliceIndex-1].numberOfInSliceEdges;
                perSliceDataVec[sliceIndex].inSliceEdgeOffset =  prevOffset + prevEdgeNum;

                NIFTY_CHECK_OP(perSliceDataVec[sliceIndex-1].maxInSliceNode + 1, == , perSliceDataVec[sliceIndex].minInSliceNode,
                    "unusable supervoxels for GridRagStacked2D");
            }
            const auto & lastSlice =  perSliceDataVec.back();
            rag.numberOfInSliceEdges_ = lastSlice.inSliceEdgeOffset + lastSlice.numberOfInSliceEdges;
        }

        std::cout<<"phase 3\n";
        /////////////////////////////////////////////////////
        // Phase 3 : set up in slice edge indices
        /////////////////////////////////////////////////////
        {
            // temp. resize the edge vec
            auto & edges = rag.edges_;
            edges.resize(rag.numberOfInSliceEdges_);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                auto edgeIndex = sliceData.inSliceEdgeOffset;
                const auto startNode = sliceData.minInSliceNode;
                const auto endNode = sliceData.maxInSliceNode+1;
                
                for(uint64_t u = startNode; u< endNode; ++u){
                    for(auto & vAdj :  rag.nodes_[u]){
                        const auto v = vAdj.node();
                        if(u < v){
                            edges[edgeIndex] = typename RagType::EdgeStorage(u, v);
                            vAdj.changeEdgeIndex(edgeIndex);
                            auto fres =  rag.nodes_[v].find(typename RagType::NodeAdjacency(u));
                            fres->changeEdgeIndex(edgeIndex);
                            ++edgeIndex;
                        }
                    }
                }
                NIFTY_CHECK_OP(edgeIndex, ==, sliceData.inSliceEdgeOffset + sliceData.numberOfInSliceEdges,"");
            });
        }

        std::cout<<"phase 4\n";
        /////////////////////////////////////////////////////
        // Phase 4 : between slice edges
        /////////////////////////////////////////////////////
        {

            const Coord sliceABShape({int64_t(2),shape[1], shape[2]});
            BlockStorageType sliceABStorage(threadpool, sliceABShape, nThreads);

            for(auto startIndex : {0,1}){
                parallel::parallel_foreach(threadpool, numberOfSlices-1, [&](const int tid, const int64_t sliceAIndex){

                    // this seems super ugly...
                    // there must be a better way to loop in parallel 
                    // over first the odd then the even coordinates
                    const auto oddIndex = bool(sliceAIndex%2);
                    if(startIndex==0 && !oddIndex || startIndex==1 && oddIndex ){

                        const auto sliceBIndex = sliceAIndex + 1;

                        // fetch the data for the slice
                        const Coord blockABBegin({sliceAIndex,int64_t(0),int64_t(0)});
                        const Coord blockABEnd({sliceAIndex+2, sliceShape2[0], sliceShape2[1]});
                        //auto & sliceAB  = perThreadDataVec[tid].sliceAB;

                        auto sliceAB = sliceABStorage.getView(tid);

                        labelsProxy.readSubarray(blockABBegin, blockABEnd, sliceAB);
                        const Coord coordAOffset{0L,0L,0L};
                        const Coord coordBOffset{1L,0L,0L};
                        auto sliceALabels = sliceAB.view(coordAOffset.begin(), sliceShape3.begin()).squeezedView();
                        auto sliceBLabels = sliceAB.view(coordBOffset.begin(), sliceShape3.begin()).squeezedView();

                        nifty::tools::forEachCoordinate(sliceShape2,[&](const Coord2 & coord){
                            const auto lU = sliceALabels(coord.asStdArray());
                            const auto lV = sliceBLabels(coord.asStdArray());
                            if(rag.insertEdgeOnlyInNodeAdj(lU, lV)){
                                ++perSliceDataVec[sliceAIndex].numberOfToNextSliceEdges;
                            }                      
                        });
                    }
                });
            }
        }

        std::cout<<"phase 5\n";
        /////////////////////////////////////////////////////
        // Phase 5 : set up the between slice edge offsets
        /////////////////////////////////////////////////////
        {
            rag.numberOfInBetweenSliceEdges_ += perSliceDataVec[0].numberOfToNextSliceEdges;
            perSliceDataVec[0].toNextSliceEdgeOffset = rag.numberOfInSliceEdges_;
            for(auto sliceIndex=1; sliceIndex<numberOfSlices; ++sliceIndex){
                const auto prevOffset = perSliceDataVec[sliceIndex-1].toNextSliceEdgeOffset;
                const auto prevEdgeNum = perSliceDataVec[sliceIndex-1].numberOfToNextSliceEdges;
                perSliceDataVec[sliceIndex].toNextSliceEdgeOffset =  prevOffset + prevEdgeNum;

                rag.numberOfInBetweenSliceEdges_ +=  perSliceDataVec[sliceIndex].numberOfToNextSliceEdges;
            }
            const auto & lastSlice =  perSliceDataVec.back();
        }

        std::cout<<"phase 6\n";
        /////////////////////////////////////////////////////
        // Phase 6 : set up between slice edge indices
        /////////////////////////////////////////////////////
        {
            // temp. resize the edge vec
            auto & edges = rag.edges_;
            edges.resize(rag.numberOfInSliceEdges_ + rag.numberOfInBetweenSliceEdges_);

            parallel::parallel_foreach(threadpool, numberOfSlices, [&](const int tid, const int64_t sliceIndex){
                auto & sliceData  = perSliceDataVec[sliceIndex];

                auto edgeIndex = sliceData.toNextSliceEdgeOffset;
                const auto startNode = sliceData.minInSliceNode;
                const auto endNode = sliceData.maxInSliceNode+1;
                
                for(uint64_t u = startNode; u< endNode; ++u){
                    for(auto & vAdj :  rag.nodes_[u]){
                        const auto v = vAdj.node();
                        if(u < v && v >= endNode){
                            edges[edgeIndex] = typename RagType::EdgeStorage(u, v);
                            vAdj.changeEdgeIndex(edgeIndex);
                            auto fres =  rag.nodes_[v].find(typename RagType::NodeAdjacency(u));
                            fres->changeEdgeIndex(edgeIndex);
                            ++edgeIndex;
                        }
                    }
                }
                //NIFTY_CHECK_OP(edgeIndex, ==, sliceData.inSliceEdgeOffset + sliceData.numberOfInSliceEdges,"");
            });
        }
    } 
};







} // end namespace detail_rag
} // end namespace graph
} // end namespace nifty


#endif /* NIFTY_GRAPH_RAG_DETAIL_RAG_COMPUTE_GRID_RAG_HXX */
