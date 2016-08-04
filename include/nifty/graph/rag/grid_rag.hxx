#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_HXX


#include <random>
#include <functional>
#include <ctime>
#include <stack>
#include <algorithm>

// for strange reason travis does not find the boost flat set
#ifdef WITHIN_TRAVIS
#include <set>
#define __setimpl std::set
#else
#include <boost/container/flat_set.hpp>
#define __setimpl boost::container::flat_set
#endif

//#include <parallel/algorithm>
#include <unordered_set>

#include "nifty/array/arithmetic_array.hxx"
#include "nifty/graph/rag/grid_rag_labels.hxx"
#include "nifty/graph/rag/detail_rag/compute_grid_rag.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/graph/undirected_list_graph.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/timer.hxx"
//#include "nifty/graph/detail/contiguous_indices.hxx"


namespace nifty{
namespace graph{


template<size_t DIM, class LABEL_TYPE>
class ExplicitLabels;


template<size_t DIM, class LABELS_PROXY>
class GridRag : public UndirectedGraph<>{
public:
    typedef LABELS_PROXY LabelsProxy;
    struct Settings{
        Settings()
        :   numberOfThreads(-1),
            blockShape()
        {
            for(auto d=0; d<DIM; ++d)
                blockShape[d] = 100;
        }
        int numberOfThreads;
        array::StaticArray<int64_t, DIM> blockShape;
    };

    typedef GridRag<DIM, LABELS_PROXY> SelfType;

    friend class detail_rag::ComputeRag< SelfType >;


    GridRag(const LabelsProxy & labelsProxy, const Settings & settings = Settings())
    :   settings_(settings),
        labelsProxy_(labelsProxy)
    {
        detail_rag::ComputeRag< SelfType >::computeRag(*this, settings_);
    }

    const LabelsProxy & labelsProxy() const {
        return labelsProxy_;
    }
private:
    Settings settings_;
    LabelsProxy labelsProxy_;

};


template<unsigned int DIM, class LABEL_TYPE>
using ExplicitLabelsGridRag = GridRag<DIM, ExplicitLabels<DIM, LABEL_TYPE> > ; 


} // end namespace graph
} // end namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_HXX */
