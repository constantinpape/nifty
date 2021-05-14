#include <pybind11/pybind11.h>
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/agglo/export_agglomerative_clustering.hxx"
#include "nifty/graph/graph_maps.hxx"
#include "nifty/graph/agglo/agglomerative_clustering.hxx"


#include "nifty/graph/agglo/cluster_policies/node_weighted_cluster_policy.hxx"
#include "nifty/graph/agglo/cluster_policies/detail/node_merge_rules.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace agglo{

    template<class G, class UPDATE_RULE, class DISTANCE_FUNCTION, bool WITH_UCM>
    void exportNodeWeightedClusterPolicy(py::module & m) {

        typedef G GraphType;
        const auto graphName = GraphName<GraphType>::name();
        typedef xt::pytensor<float, 2> PyNodeFeatures;
        typedef xt::pytensor<float, 1> PyViewFloat;
        const std::string withUcmStr = WITH_UCM ? std::string("WithUcm") : std::string();

        // name and type of cluster operator
        typedef NodeWeightedClusterPolicy<
            GraphType, UPDATE_RULE, DISTANCE_FUNCTION, WITH_UCM
        > ClusterPolicyType;
        const auto baseName = std::string("NodeWeightedClusterPolicy") + UPDATE_RULE::staticName() +
                              DISTANCE_FUNCTION::staticName() + withUcmStr;
        const auto clsName = baseName + graphName;
        const auto facName = lowerFirst(baseName);

        auto cls = py::class_<ClusterPolicyType>(m, clsName.c_str());

        // the cluster operator cls
        m.def(facName.c_str(),
            [](const GraphType & graph,
               const PyNodeFeatures & nodeFeatures,
               const PyViewFloat & nodeSizes,
               const typename ClusterPolicyType::DistanceSettingsType distanceSettings,
               const double threshold,
               const uint64_t numberOfNodesStop,
               const double sizeRegularizer){

                typename ClusterPolicyType::SettingsType s;
                s.distanceSettings = distanceSettings;
                s.threshold = threshold;
                s.numberOfNodesStop = numberOfNodesStop;
                s.sizeRegularizer = sizeRegularizer;

                auto ptr = new ClusterPolicyType(
                        graph, nodeFeatures, nodeSizes, s
                );
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(), // graph
            py::arg("graph"),
            py::arg("nodeFeatures"),
            py::arg("nodeSizes"),
            py::arg("distanceSettings"),
            py::arg("threshold")=0.0,
            py::arg("numberOfNodesStop")=1,
            py::arg("sizeRegularizer")=0.0
        );

        // export the agglomerative clustering functionality for this cluster operator
        exportAgglomerativeClusteringTClusterPolicy<ClusterPolicyType>(m, baseName);
    }

    template<class G>
    void exportNodeWeightedClusteringT(py::module & m) {

        // merge rules
        typedef merge_rules::ArithmeticMeanMergeRule<G, float> mean;
        typedef merge_rules::MaxMergeRule<G, float> max;
        typedef merge_rules::MinMergeRule<G, float> min;

        // distance functions
        typedef merge_rules::L1Distance l1;
        typedef merge_rules::L2Distance l2;
        typedef merge_rules::CosineDistance cosine;

        // with mean merge rule
        exportNodeWeightedClusterPolicy<G, mean, l1, false>(m);
        exportNodeWeightedClusterPolicy<G, mean, l2, false>(m);
        exportNodeWeightedClusterPolicy<G, mean, cosine, false>(m);
        
        // with max merge rule
        exportNodeWeightedClusterPolicy<G, max, l1, false>(m);
        exportNodeWeightedClusterPolicy<G, max, l2, false>(m);
        exportNodeWeightedClusterPolicy<G, max, cosine, false>(m);
        
        // with min merge rule
        exportNodeWeightedClusterPolicy<G, min, l1, false>(m);
        exportNodeWeightedClusterPolicy<G, min, l2, false>(m);
        exportNodeWeightedClusterPolicy<G, min, cosine, false>(m);
    }

    void exportDistanceSettings(py::module & m) {
        py::class_<merge_rules::DistanceSettings>(m, "DistanceSettings")
            .def(py::init<double, bool, double>(),
               py::arg("delta")=0.0,
               py::arg("signedWeights")=false,
               py::arg("beta")=0.5
            )
        ;
    }

    void exportNodeWeightedClustering(py::module & m) {
        exportDistanceSettings(m);
        exportNodeWeightedClusteringT<PyUndirectedGraph>(m);
        exportNodeWeightedClusteringT<UndirectedGridGraph<2, true>>(m);
        exportNodeWeightedClusteringT<UndirectedGridGraph<3, true>>(m);
    }

} // namespace agglo
} // namespace graph
} // namespace nifty
