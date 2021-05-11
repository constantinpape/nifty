#pragma once

#include <string>
#include <nifty/histogram/histogram.hxx>
#include <cmath>

#include "nifty/tools/runtime_check.hxx"
#include <nifty/nifty.hxx>


namespace nifty{
namespace graph{
namespace agglo{
namespace merge_rules{

    //
    // node merge rule base class and merge rule implementations
    //

    template<class G, class T>
    class NodeMergeRule {
    public:
        typedef G GraphType;
        typedef xt::xtensor<T, 2> NodeFeaturesType;
        typedef typename GraphType:: template NodeMap<T> NodeSizesType;

        template<class NODE_FEATURES, class NODE_SIZES>
        NodeMergeRule(
            const GraphType & g,
            const NODE_FEATURES & nodeFeatures,
            const NODE_SIZES & nodeSizes
        ) : nodeFeatures_({nodeFeatures.shape()[0], nodeFeatures.shape()[1]}),
            nodeSizes_(g)
        {
            const auto nFeats = nodeFeatures.shape()[1];
            g.forEachNode([&](const uint64_t node) {
                nodeSizes_[node] = nodeSizes[node];
                for(auto featId = 0; featId < nFeats; ++featId){
                    nodeFeatures_(node, featId) = nodeFeatures(node, featId);
                }
            });
        }

        const NodeFeaturesType & nodeFeatures() const {
            return nodeFeatures_;
        }

        const NodeSizesType & nodeSizes() const {
            return nodeSizes_;
        }

    protected:
        NodeFeaturesType nodeFeatures_;
        NodeSizesType nodeSizes_;
    };


    template<class G, class T>
    class ArithmeticMeanMergeRule: public NodeMergeRule<G, T> {
    public:

        template<class NODE_FEATURES, class NODE_SIZES>
        ArithmeticMeanMergeRule(
            const G & g,
            const NODE_FEATURES & nodeFeatures,
            const NODE_SIZES & nodeSizes) : NodeMergeRule<G, T>(g, nodeFeatures, nodeSizes)
        {}

        inline void merge(const uint64_t aliveNode, const uint64_t deadNode) {

            auto & nodeSizes = this->nodeSizes_;
            auto & nodeFeatures = this->nodeFeatures_;

            auto & size = nodeSizes[aliveNode];
            const auto deadSize = nodeSizes[deadNode];
            const auto totalSize = size + deadSize;

            for(int i = 0; i < nodeFeatures.shape()[1]; ++i) {
                nodeFeatures(aliveNode, i) = (nodeFeatures(aliveNode, i) * size + nodeFeatures(deadNode) * deadSize) / totalSize;
            }
            size += deadSize;
        }

        static auto staticName(){
            return std::string("Mean");
        }
    };

    template<class G, class T>
    class MaxMergeRule: public NodeMergeRule<G, T> {
    public:

        template<class NODE_FEATURES, class NODE_SIZES>
        MaxMergeRule(
            const G & g,
            const NODE_FEATURES & nodeFeatures,
            const NODE_SIZES & nodeSizes) : NodeMergeRule<G, T>(g, nodeFeatures, nodeSizes)
        {}

        inline void merge(const uint64_t aliveNode, const uint64_t deadNode) {

            auto & nodeSizes = this->nodeSizes_;
            auto & nodeFeatures = this->nodeFeatures_;
            nodeSizes[aliveNode] += nodeSizes[deadNode];
            for(int i = 0; i < nodeFeatures.shape()[1]; ++i) {
                nodeFeatures(aliveNode, i) = std::max(nodeFeatures(aliveNode, i), nodeFeatures(deadNode));
            }
        }

        static auto staticName(){
            return std::string("Max");
        }
    };

    template<class G, class T>
    class MinMergeRule: public NodeMergeRule<G, T> {
    public:

        template<class NODE_FEATURES, class NODE_SIZES>
        MinMergeRule(
            const G & g,
            const NODE_FEATURES & nodeFeatures,
            const NODE_SIZES & nodeSizes) : NodeMergeRule<G, T>(g, nodeFeatures, nodeSizes)
        {}

        inline void merge(const uint64_t aliveNode, const uint64_t deadNode) {

            auto & nodeSizes = this->nodeSizes_;
            auto & nodeFeatures = this->nodeFeatures_;
            nodeSizes[aliveNode] += nodeSizes[deadNode];
            for(int i = 0; i < nodeFeatures.shape()[1]; ++i) {
                nodeFeatures(aliveNode, i) = std::min(nodeFeatures(aliveNode, i), nodeFeatures(deadNode));
            }
        }

        static auto staticName(){
            return std::string("Min");
        }
    };

    // TODO do we implement normalisation for the distance functions?
    // TODO do we implement casting to signed weights?
    //
    // distance function implementations
    //
    struct DistanceSettings {
        double delta{0.0};
    };

    class Distance {
    public:
        typedef DistanceSettings SettingsType;

        Distance(const SettingsType & settings): settings_(settings){
        }

    protected:
        inline double postprocessDistance(double dist) const {
            const double delta = settings_.delta;
            if(delta > 0.0) {
                dist = (2 * delta - dist) / (2 * delta);
                dist = 1. - std::max(dist, 0.) * std::max(dist, 0.0);
            }
            return dist;
        }

    protected:
        SettingsType settings_;
    };

    class L1Distance: public Distance{
    public:
        typedef Distance::SettingsType SettingsType;
        L1Distance(const SettingsType & settings): Distance(settings) {
        }

        template<class NODE_FEATURES>
        inline double operator()(const NODE_FEATURES & nodeFeats, const uint64_t u, const uint64_t v) const {
            double ret = 0.0;
            for(unsigned i = 0; i < nodeFeats.shape()[1]; ++i) {
                ret += std::abs(nodeFeats(u, i) - nodeFeats(v, i));
            }
            ret = postprocessDistance(ret);
            return ret;
        }

        static auto staticName(){
            return std::string("L1");
        }
    };

    class L2Distance: public Distance{
    public:
        typedef Distance::SettingsType SettingsType;
        L2Distance(const SettingsType & settings): Distance(settings) {
        }

        template<class NODE_FEATURES>
        inline double operator()(const NODE_FEATURES & nodeFeats, const uint64_t u, const uint64_t v) const {
            double ret = 0.0;
            double fU, fV;
            for(unsigned i = 0; i < nodeFeats.shape()[1]; ++i) {
                fU = nodeFeats(u, i);
                fV = nodeFeats(v, i);
                ret += (fU - fV) * (fU - fV);
            }
            ret = std::sqrt(ret);
            ret = postprocessDistance(ret);
            return ret;
        }

        static auto staticName(){
            return std::string("L2");
        }
    };

    class CosineDistance: public Distance{
    public:
        typedef Distance::SettingsType SettingsType;
        CosineDistance(const SettingsType & settings): Distance(settings) {
        }

        template<class NODE_FEATURES>
        inline double operator()(const NODE_FEATURES & nodeFeats, const uint64_t u, const uint64_t v) const {
            const double eps = 1e-7;

            double ret = 0.0;
            double normU = 0;
            double normV = 0;

            double fU, fV;
            for(unsigned i = 0; i < nodeFeats.shape()[1]; ++i) {
                fU = nodeFeats(u, i);
                fV = nodeFeats(v, i);
                ret += fU * fV;
                normU += fU * fU;
                normV += fV * fV;
            }
            normU = std::sqrt(normU) + eps;
            normV = std::sqrt(normV) + eps;
            ret = 1. - (ret / normU / normV);
            ret = postprocessDistance(ret);
            return ret;
        }

        static auto staticName(){
            return std::string("Cosine");
        }
    };
}
}
}
}
