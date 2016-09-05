#pragma once
#ifndef NIFTY_PYTHON_GRAPH_MULTICUT_MULTICUT_VISITOR_BASE_HXX
#define NIFTY_PYTHON_GRAPH_MULTICUT_MULTICUT_VISITOR_BASE_HXX

#include <string>
#include <initializer_list>

#include "nifty/graph/optimization/multicut/multicut_base.hxx"

namespace nifty {
namespace graph {








template<class OBJECTIVE>
class PyMulticutVisitorBase : public MulticutVisitorBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MulticutFactory<Objective>::MulticutFactory;

    typedef OBJECTIVE Objective;
    typedef MulticutVisitorBase<OBJECTIVE> VisitorBase;
    typedef MulticutBase<Objective> McBase;
    typedef typename Objective::Graph Graph;



    void begin(McBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            void,                       /* Return type */
            VisitorBase,                /* Parent class */
            begin,                      /* Name of function */
            mcBase                      /* Argument(s) */
        );
    }

    bool visit(McBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            void,                       /* Return type */
            VisitorBase,                /* Parent class */
            visit,                      /* Name of function */
            mcBase                      /* Argument(s) */
        );
    }

    void end(McBase * mcBase) {
        PYBIND11_OVERLOAD_PURE(
            void,                    /* Return type */
            VisitorBase,             /* Parent class */
            end,                     /* Name of function */
            mcBase                   /* Argument(s) */
        );
    }

    void addLogNames(std::initializer_list<std::string> logNames) {
        PYBIND11_OVERLOAD(
            void,                    /* Return type */
            VisitorBase,             /* Parent class */
            addLogNames,                     /* Name of function */
            logNames                   /* Argument(s) */
        );
    }

    void setLogValue(const size_t logIndex, double logValue) {
        PYBIND11_OVERLOAD(
            void,                    /* Return type */
            VisitorBase,             /* Parent class */
            setLogValue,                     /* Name of function */
            logIndex,logValue        /* Argument(s) */
        );
    }

};


} // namespace graph
} // namespace nifty

#endif /* NIFTY_PYTHON_GRAPH_MULTICUT_MULTICUT_VISITOR_BASE_HXX */
