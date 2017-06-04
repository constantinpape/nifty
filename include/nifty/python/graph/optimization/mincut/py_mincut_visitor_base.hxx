#pragma once

#include <string>
#include <initializer_list>

#include "nifty/graph/optimization/mincut/mincut_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace mincut{






template<class OBJECTIVE>
class PyMincutVisitorBase : public MincutVisitorBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MincutFactory<Objective>::MincutFactory;

    typedef OBJECTIVE Objective;
    typedef MincutVisitorBase<OBJECTIVE> VisitorBase;
    typedef MincutBase<Objective> McBase;
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
            bool,                       /* Return type */
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

} // namespace mincut
} // namespace optimization
} // namespace graph
} // namespace nifty

