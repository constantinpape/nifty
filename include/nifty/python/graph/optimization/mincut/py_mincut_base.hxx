#pragma once

#include "nifty/graph/optimization/mincut/mincut_base.hxx"

namespace nifty {
namespace graph {
namespace optimization{
namespace mincut{






template<class OBJECTIVE>
class PyMincutBase : public MincutBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using MincutFactory<Objective>::MincutFactory;

    typedef OBJECTIVE Objective;
    typedef MincutVisitorBase<OBJECTIVE> VisitorBase;
    typedef MincutBase<Objective> McBase;
    typedef typename Objective::Graph Graph;
    typedef typename McBase::EdgeLabels EdgeLabels;
    typedef typename McBase::NodeLabels NodeLabels;


    /* Trampoline (need one for each virtual function) */
    void optimize(NodeLabels & nodeLabels, VisitorBase * visitor) {
        PYBIND11_OVERLOAD_PURE(
            void,                  /* Return type */
            McBase,                /* Parent class */
            optimize,              /* Name of function */
            nodeLabels,  visitor   /* Argument(s) */
        );
    }

    const NodeLabels & currentBestNodeLabels()  {
        PYBIND11_OVERLOAD_PURE(
            const NodeLabels &,                 /* Return type */
            McBase,                             /* Parent class */
            currentBestNodeLabels,              /* Name of function */
        );
    }

    double currentBestEnergy()  {
        PYBIND11_OVERLOAD_PURE(
            double,                  /* Return type */
            McBase,                  /* Parent class */
            currentBestEnergy,       /* Name of function */
        );
    }

    const Objective & objective() const {
        PYBIND11_OVERLOAD_PURE(
            const Objective & ,    /* Return type */
            McBase,                /* Parent class */
            objective              /* Name of function */
        );
    }

    std::string name() const{
        PYBIND11_OVERLOAD_PURE(
            std::string ,    /* Return type */
            McBase,          /* Parent class */
            name             /* Name of function */
        );
    }
};

} // namespace mincut
} // namespace optimization
} // namespace graph
} // namespace nifty

