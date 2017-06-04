#pragma once

#include <cstddef>
#include <utility> // std::pair
#include <queue>

#include "nifty/graph/subgraph_mask.hxx" // DefaultSubgraphMask

namespace nifty {
namespace graph {

/// Search a path for a chord.
///
/// \param graph Graph.
/// \param begin Iterator to the beginning of the sequence of nodes on the path.
/// \param end Iterator to the end of the sequence of nodes on the path.
/// \param ignoreEdgeBetweenFirstAndLast Flag.
///
template<class GRAPH, class ITERATOR>
inline int64_t
findChord(
    const GRAPH& graph,
    ITERATOR begin,
    ITERATOR end,
    const bool ignoreEdgeBetweenFirstAndLast = false
) {
    return findChord(graph, DefaultSubgraphMask<GRAPH>(), begin, end, 
        ignoreEdgeBetweenFirstAndLast);
}



/// Search a path for a chord.
///
/// \param graph Graph.
/// \param mask A subgraph mask such as DefaultSubgraphMask.
/// \param begin Iterator to the beginning of the sequence of nodes on the path.
/// \param end Iterator to the end of the sequence of nodes on the path.
/// \param ignoreEdgeBetweenFirstAndLast Flag.
///
template<class GRAPH, class SUBGRAPH_MASK, class ITERATOR>
inline int64_t
findChord(
    const GRAPH& graph,
    const SUBGRAPH_MASK& mask,
    ITERATOR begin,
    ITERATOR end,
    const bool ignoreEdgeBetweenFirstAndLast = false
) {
    //std::cout<<"path size "<<std::distance(begin, end)<<"\n";
    //std::cout<<" * \n";
    for(ITERATOR it = begin; it != end - 1; ++it){
        //std::cout<<" *** \n";
        for(ITERATOR it2 = it + 2; it2 != end; ++it2) {
            if(ignoreEdgeBetweenFirstAndLast && it == begin && it2 == end - 1) {
                continue;
            }
            //std::cout<<" findEdge "<<*it<<" "<<*it2<<" \n";

            const auto p = graph.findEdge(*it, *it2);
            if(p !=- 1 && mask.useEdge(p)) {
                return p;
            }
        }
    }
    return -1;
}


template<class PREDECESSORS_MAP, class OUT_ITER>
size_t buildPathInLargeEnoughBuffer(
    const uint64_t source,
    const uint64_t target,
    const PREDECESSORS_MAP & predecessorMap,
    OUT_ITER  largeEnoughBufferBegin
){
    auto current = target;
    *largeEnoughBufferBegin = current;
    ++largeEnoughBufferBegin;
    uint64_t c = 1;
    while(current != source){
        current = predecessorMap[current];
        *largeEnoughBufferBegin = current;
        ++largeEnoughBufferBegin;
        ++c;
    }
    return c;
}


} // namespace graph
} // namespace nifty

