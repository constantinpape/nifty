#pragma once

#include "nifty/tools/changable_priority_queue.hxx"
#include "nifty/graph/edge_contraction_graph.hxx"
#include "nifty/graph/agglo/cluster_policies/cluster_policies_common.hxx"

namespace nifty{
namespace graph{
namespace agglo{

    // UPDATE_RULE: the type of linkage criteria implemented in ./details/node_merge_rules.hxx
    // DISTANCE_FUNCTION: the distance used to compute edge weights from node features
    // UCM: ultra contour map (use an edge union-find datastructure)
    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    class NodeWeightedClusterPolicy {
    public:
        // typedefs
        typedef NodeWeightedClusterPolicy<
            GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM
        > SelfType;
        typedef GRAPH GraphType;
        typedef EdgeContractionGraph<GraphType, SelfType> EdgeContractionGraphType;
        typedef nifty::tools::ChangeablePriorityQueue<float, std::greater<float>> QueueType;
        typedef typename GRAPH:: template EdgeMap<float> EdgeMapType;
        typedef typename DISTANCE_FUNCTION::SettingsType DistanceSettingsType;

        struct SettingsType{
            DistanceSettingsType distanceSettings;
            uint64_t numberOfNodesStop{1};
            double sizeRegularizer{0.};
            double threshold{0.};
        };

    public:
        // constructor
        template<class NODE_FEATURES, class NODE_SIZES>
        NodeWeightedClusterPolicy(
            const GraphType & graph,
            const NODE_FEATURES & nodeFeatures,
            const NODE_SIZES & nodeSizes,
            const SettingsType & settings
        );

        // edge contraction graph callbacks
        void contractEdge(const uint64_t edgeToContract);
        void mergeNodes(const uint64_t aliveNode, const uint64_t deadNode);
        void mergeEdges(const uint64_t aliveEdge, const uint64_t deadEdge);
        void contractEdgeDone(const uint64_t edgeToContract);

        // API called in agglomerative clustering
        std::pair<uint64_t, double> edgeToContractNext() const;
        bool isDone();
        EdgeContractionGraphType & edgeContractionGraph();

    // private functions
    private:
        double pqMergePrio(const uint64_t edge) const;

        inline double computeWeight(const uint64_t edge) const {
            const auto & nodeFeats = nodeWeights_.nodeFeatures();

            const auto u = edgeContractionGraph_.u(edge);
            const auto v = edgeContractionGraph_.v(edge);

            double w = distanceFunction_(nodeFeats, u, v);

            const auto sr = settings_.sizeRegularizer;
            if(sr > 0.0) {
                const auto & nodeSizes = nodeWeights_.nodeSizes();
                const double sizeU = nodeSizes[u];
                const double sizeV = nodeSizes[v];
                const double regularizer = 2.0 / (1.0 / std::pow(sizeU,sr) + 1.0 / std::pow(sizeV,sr));
                w *= regularizer;
            }
            return w;
        }

    // private members
    private:
        const GraphType & graph_;
        EdgeContractionGraphType edgeContractionGraph_;
        QueueType pq_;
        SettingsType settings_;

        UPDATE_RULE nodeWeights_;
        DISTANCE_FUNCTION distanceFunction_;

        uint64_t edgeToContractNext_;
        double edgeToContractNextMergePrio_;
    };

    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    template<class NODE_FEATURES, class NODE_SIZES>
    NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::
    NodeWeightedClusterPolicy(
        const GraphType & graph,
        const NODE_FEATURES & nodeFeatures,
        const NODE_SIZES & nodeSizes,
        const SettingsType & settings
    ) : graph_(graph),
        edgeContractionGraph_(graph, *this),
        pq_(graph.edgeIdUpperBound()+1),
        settings_(settings),
        nodeWeights_(graph, nodeFeatures, nodeSizes),
        distanceFunction_(settings.distanceSettings)
    {
        graph_.forEachEdge([&](const uint64_t edge){
            pq_.push(edge, this->computeWeight(edge));
        });
    }

    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    inline void
    NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::
    contractEdge(
        const uint64_t edgeToContract
    ){
        pq_.deleteItem(edgeToContract);
    }

    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    inline typename NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::EdgeContractionGraphType &
    NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::
    edgeContractionGraph(){
        return edgeContractionGraph_;
    }

    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    inline void
    NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::
    mergeNodes(
        const uint64_t aliveNode,
        const uint64_t deadNode
    ){
        // merge node sizes and node weights
        nodeWeights_.merge(aliveNode, deadNode);
    }

    // TODO
    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    inline void
    NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::
    mergeEdges(
        const uint64_t aliveEdge,
        const uint64_t deadEdge
    ){
        pq_.deleteItem(deadEdge);
    }

    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    inline void
    NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::
    contractEdgeDone(
        const uint64_t edgeToContract
    ){
        const auto u = edgeContractionGraph_.nodeOfDeadEdge(edgeToContract);
        for(auto adj : edgeContractionGraph_.adjacency(u)){
            const auto edge = adj.edge();
            pq_.push(edge, computeWeight(edge));
        }
    }

    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    inline bool
    NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::isDone(
    ){
        if(edgeContractionGraph_.numberOfNodes() <= settings_.numberOfNodesStop) {
            return true;
        }
        if(pq_.topPriority() <= settings_.threshold) {
            return true;
        }
        return false;
    }

    template<class GRAPH, class UPDATE_RULE, class DISTANCE_FUNCTION, bool ENABLE_UCM>
    inline std::pair<uint64_t, double>
    NodeWeightedClusterPolicy<GRAPH, UPDATE_RULE, DISTANCE_FUNCTION, ENABLE_UCM>::
    edgeToContractNext() const {
        return std::pair<uint64_t, double>(pq_.top(), pq_.topPriority()) ;
    }

} // namespace agglo
} // namespace graph
} // namespace nifty
