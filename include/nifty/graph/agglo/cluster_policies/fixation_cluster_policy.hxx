#pragma once

#include <functional>
#include <set>
#include <unordered_set>
#include <boost/container/flat_set.hpp>

#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"



namespace nifty{
namespace graph{
namespace agglo{


template<
    class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM
>
class FixationClusterPolicy{

    typedef FixationClusterPolicy<
        GRAPH, ACC_0, ACC_1, ENABLE_UCM
    > SelfType;

private:    
    typedef typename GRAPH:: template EdgeMap<uint8_t> UInt8EdgeMap;
    typedef typename GRAPH:: template EdgeMap<float> FloatEdgeMap;
    typedef typename GRAPH:: template NodeMap<float> FloatNodeMap;


    typedef ACC_0 Acc0Type;
    typedef ACC_1 Acc1Type;
public:
    typedef typename Acc0Type::SettingsType Acc0SettingsType;
    typedef typename Acc1Type::SettingsType Acc1SettingsType;


    // input types
    typedef GRAPH                                       GraphType;
    typedef FloatEdgeMap                                EdgePrioType;
    typedef FloatEdgeMap                                EdgeSizesType;
    typedef FloatNodeMap                                NodeSizesType;

    struct SettingsType{

     
        Acc0SettingsType updateRule0;
        Acc1SettingsType updateRule1;
        bool zeroInit = false;
        uint64_t numberOfNodesStop{1};
        //uint64_t numberOfBins{40};
    };

    enum class EdgeStates : uint8_t { 
        PURE_LOCAL = 0, 
        LOCAL = 1,
        LIFTED = 2,
        PURE_LIFTED = 3
    };



    typedef EdgeContractionGraph<GraphType, SelfType>   EdgeContractionGraphType;

    friend class EdgeContractionGraph<GraphType, SelfType, ENABLE_UCM> ;
private:

    // internal types


    typedef nifty::tools::ChangeablePriorityQueue< float , std::greater<float> > QueueType;

public:

    template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_LOCAL_EDGE, class EDGE_SIZES>
    FixationClusterPolicy(const GraphType &, 
                              const MERGE_PRIOS & , 
                              const NOT_MERGE_PRIOS &,
                              const IS_LOCAL_EDGE &,
                              const EDGE_SIZES & , 
                              const SettingsType & settings = SettingsType());


    std::pair<uint64_t, double> edgeToContractNext() const;
    bool isDone();

    // callback called by edge contraction graph

    EdgeContractionGraphType & edgeContractionGraph();

private:
    double pqMergePrio(const uint64_t edge) const;

public:
    // callbacks called by edge contraction graph
    void contractEdge(const uint64_t edgeToContract);
    void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
    void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
    void contractEdgeDone(const uint64_t edgeToContract);

    bool isMergeAllowed(const uint64_t edge){
        const auto  s = edgeState_[edge];
        if(s == EdgeStates::PURE_LOCAL){
            return true;
        }
        else if(s == EdgeStates::LOCAL){
            return acc0_[edge] > acc1_[edge];
        }
        else{
            return false;
        }
    }

private:



    // INPUT
    const GraphType &   graph_;


    ACC_0 acc0_;
    ACC_1 acc1_;

    typename GRAPH:: template EdgeMap<EdgeStates>  edgeState_;

    SettingsType        settings_;
    
    // INTERNAL
    EdgeContractionGraphType edgeContractionGraph_;
    QueueType pq_;

    uint64_t edgeToContractNext_;
    double   edgeToContractNextMergePrio_;
};


template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
template<class MERGE_PRIOS, class NOT_MERGE_PRIOS, class IS_LOCAL_EDGE,class EDGE_SIZES>
inline FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
FixationClusterPolicy(
    const GraphType & graph,
    const MERGE_PRIOS & mergePrios,
    const NOT_MERGE_PRIOS & notMergePrios,
    const IS_LOCAL_EDGE & isLocalEdge,
    const EDGE_SIZES      & edgeSizes,
    const SettingsType & settings
)
:   graph_(graph),
    acc0_(graph, mergePrios,    edgeSizes, settings.updateRule0),
    acc1_(graph, notMergePrios, edgeSizes, settings.updateRule1),
    edgeState_(graph),
    pq_(graph.edgeIdUpperBound()+1),
    settings_(settings),
    edgeContractionGraph_(graph, *this)
{
   
    graph_.forEachEdge([&](const uint64_t edge){

        const auto loc = isLocalEdge[edge];

        edgeState_[edge] = (loc ? EdgeStates::PURE_LOCAL : EdgeStates::PURE_LIFTED);

        if(settings_.zeroInit){
            if(loc){
                acc1_.set(edge, 0.0, edgeSizes[edge]);
            }
            else{
                acc0_.set(edge, 0.0, edgeSizes[edge]);
            }
        }

        pq_.push(edge, this->pqMergePrio(edge));
    });
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline std::pair<uint64_t, double> 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
edgeToContractNext() const {    
    return std::pair<uint64_t, double>(edgeToContractNext_,edgeToContractNextMergePrio_) ;
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline bool 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::isDone(
){

    if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop ||
       pq_.empty() || isNegativeInf(pq_.topPriority())
    ){
        return  true;
    }
    else{
        while(!isNegativeInf(pq_.topPriority())){

            const auto nextActioneEdge = pq_.top();

        

            if(this->isMergeAllowed(nextActioneEdge)){
                edgeToContractNext_ = nextActioneEdge;
                edgeToContractNextMergePrio_ = pq_.topPriority();
                return false;
            }
            else{
                pq_.push(nextActioneEdge, -1.0*std::numeric_limits<double>::infinity());
            }
            
        }
        return true;
    }
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline double 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
pqMergePrio(
    const uint64_t edge
) const {
    const auto s = edgeState_[edge];
    if(s == EdgeStates::LOCAL || s==EdgeStates::PURE_LOCAL){
        return  acc0_[edge];
    }
    else{
        return -1.0*std::numeric_limits<double>::infinity(); 
    }
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline void 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
contractEdge(
    const uint64_t edgeToContract
){
    pq_.deleteItem(edgeToContract);
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline typename FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::EdgeContractionGraphType & 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
edgeContractionGraph(){
    return edgeContractionGraph_;
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline void 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
mergeNodes(
    const uint64_t aliveNode, 
    const uint64_t deadNode
){
}

template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline void 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
mergeEdges(
    const uint64_t aliveEdge, 
    const uint64_t deadEdge
){

    NIFTY_ASSERT_OP(aliveEdge,!=,deadEdge);
    NIFTY_ASSERT(pq_.contains(aliveEdge));
    NIFTY_ASSERT(pq_.contains(deadEdge));

    pq_.deleteItem(deadEdge);
   
    // update merge prio
    
    auto & sa = edgeState_[aliveEdge];
    const auto  sd = edgeState_[deadEdge];

    if(settings_.zeroInit  && sa == EdgeStates::PURE_LIFTED &&  sd != EdgeStates::PURE_LIFTED)
        acc0_.setValueFrom(aliveEdge, deadEdge);
    else
        acc0_.merge(aliveEdge, deadEdge);

    // update notMergePrio
    if(settings_.zeroInit  && sa == EdgeStates::PURE_LOCAL &&  sd !=  EdgeStates::PURE_LOCAL)
        acc1_.setValueFrom(aliveEdge, deadEdge);
    else
        acc1_.merge(aliveEdge, deadEdge);
    

    // update state
    if(sa == EdgeStates::PURE_LIFTED &&  sd == EdgeStates::PURE_LIFTED){
        sa =  EdgeStates::PURE_LIFTED;
    }
    else if(sa == EdgeStates::PURE_LOCAL &&  sd == EdgeStates::PURE_LOCAL){
        sa = EdgeStates::PURE_LOCAL;
    }
    else if(
        sa == EdgeStates::PURE_LOCAL ||  sa == EdgeStates::LOCAL ||
        sd == EdgeStates::PURE_LOCAL ||  sd == EdgeStates::LOCAL 
    ){
        sa = EdgeStates::LOCAL;
    }
    else if(sa == EdgeStates::PURE_LIFTED ||  sa == EdgeStates::LIFTED  &&
            sd == EdgeStates::PURE_LIFTED ||  sd == EdgeStates::LIFTED )
    {
        sa = EdgeStates::LIFTED;
    }


    pq_.push(aliveEdge, this->pqMergePrio(aliveEdge));
    
}


template<class GRAPH, class ACC_0, class ACC_1, bool ENABLE_UCM>
inline void 
FixationClusterPolicy<GRAPH, ACC_0, ACC_1,ENABLE_UCM>::
contractEdgeDone(
    const uint64_t edgeToContract
){
    
}


} // namespace agglo
} // namespace nifty::graph
} // namespace nifty

