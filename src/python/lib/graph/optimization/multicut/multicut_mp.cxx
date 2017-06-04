#include <pybind11/pybind11.h>

#include "nifty/graph/optimization/multicut/multicut_mp.hxx"

#include "nifty/python/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/python/converter.hxx"
#include "nifty/python/graph/optimization/multicut/export_multicut_solver.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

    template<class OBJECTIVE>
    void exportMulticutMpT(py::module & multicutModule){
        
        typedef OBJECTIVE ObjectiveType;
        typedef MulticutMp<ObjectiveType> Solver;
        typedef typename Solver::Settings Settings;

        const auto objName = MulticutObjectiveName<ObjectiveType>::name();
        const auto solverName = std::string("MulticutMp");



        ///////////////////////////////////////////////////////////////
        // DOCSTRING HELPER
        ///////////////////////////////////////////////////////////////
        nifty::graph::optimization::SolverDocstringHelper docHelper;
        docHelper.objectiveName = "multicut objective";
        docHelper.objectiveClsName = MulticutObjectiveName<OBJECTIVE>::name();
        docHelper.name = "mp-lp";
        docHelper.mainText =  
        "Optimize the multicut objective by convergent message passing as\n"
        "described by :cite:`TODO`";
        docHelper.cites.emplace_back("TODO");

        docHelper.requirements.emplace_back("WITH_MP_LP");
        docHelper.requirements.emplace_back("CPP_14");





        // FIXME verbose has no effect yet
        exportMulticutSolver<Solver>(multicutModule, solverName.c_str(), docHelper)
            .def(py::init<>())
            .def_readwrite("mcFactory",&Settings::mcFactory)
            .def_readwrite("verbose",&Settings::verbose)
            .def_readwrite("numberOfIterations",&Settings::numberOfIterations)
            .def_readwrite("primalComputationInterval",&Settings::primalComputationInterval)
            .def_readwrite("standardReparametrization",&Settings::standardReparametrization)
            .def_readwrite("roundingReparametrization",&Settings::roundingReparametrization)
            .def_readwrite("tightenReparametrization",&Settings::tightenReparametrization)
            .def_readwrite("tighten",&Settings::tighten)
            .def_readwrite("tightenInterval",&Settings::tightenInterval)
            .def_readwrite("tightenIteration",&Settings::tightenIteration)
            .def_readwrite("tightenSlope",&Settings::tightenSlope)
            .def_readwrite("tightenConstraintsPercentage",&Settings::tightenConstraintsPercentage)
            .def_readwrite("numberOfIterations",&Settings::numberOfIterations)
            .def_readwrite("minDualImprovement",&Settings::minDualImprovement)
            .def_readwrite("minDualImprovementInterval",&Settings::minDualImprovementInterval)
            .def_readwrite("timeout",&Settings::timeout)
            .def_readwrite("numberOfThreads",&Settings::numberOfThreads)
        ; 

    }

    
    void exportMulticutMp(py::module & multicutModule){
        
        {
            typedef PyUndirectedGraph GraphType;
            typedef MulticutObjective<GraphType, double> ObjectiveType;
            exportMulticutMpT<ObjectiveType>(multicutModule);
        }
        // FIXME this doesn't compile
        //{
        //    typedef PyContractionGraph<PyUndirectedGraph> GraphType;
        //    typedef MulticutObjective<GraphType, double> ObjectiveType;
        //    exportMulticutMpT<ObjectiveType>(multicutModule);
        //}     

    }

} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace graph
} // namespace nifty
