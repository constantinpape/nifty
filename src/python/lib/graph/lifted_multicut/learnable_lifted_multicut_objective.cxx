#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream> 
#include <pybind11/numpy.h>

#include "nifty/python/graph/optimization/lifted_multicut/lifted_multicut_objective.hxx"
#include "nifty/python/graph/optimization/lifted_multicut/learnable_lifted_multicut_objective.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"

#include "nifty/python/converter.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{
namespace lifted_multicut{

    template<class GRAPH>
    void exportLearnableLiftedMulticutObjectiveT(py::module & liftedMulticutModule) {

        typedef GRAPH Graph;
        typedef LiftedMulticutObjective<Graph, double> BaseObjectiveType;
        typedef LearnableLiftedMulticutObjective<Graph, float> ObjectiveType;
        typedef typename ObjectiveType::LiftedGraphType LiftedGraphType;
        const auto clsName = LiftedMulticutObjectiveName<ObjectiveType>::name();


        auto liftedMulticutObjectiveCls = py::class_<ObjectiveType>(
            liftedMulticutModule, clsName.c_str(),
            py::base<BaseObjectiveType>()
        );

    }

    void exportLearnableLiftedMulticutObjective(py::module & liftedMulticutModule) {

        {
            typedef PyUndirectedGraph GraphType;
            exportLearnableLiftedMulticutObjectiveT<GraphType>(liftedMulticutModule);
        }
    }

}
}
}
