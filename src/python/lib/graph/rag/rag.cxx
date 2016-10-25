#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{


    void exportGridRag(py::module &);
    void exportGraphAccumulator(py::module &);
    void exportProjectToPixels(py::module &);
    void exportAccumulate(py::module &);
    void exportAccumulateFromBoundingBoxes(py::module &);
    void exportAccumulateFilters(py::module &);
    void exportEdgeCoordinates(py::module &);

}
}


PYBIND11_PLUGIN(_rag) {
    py::module ragModule("_rag", "rag submodule of nifty.graph");

    using namespace nifty::graph;

    exportGridRag(ragModule);
    exportGraphAccumulator(ragModule);
    exportProjectToPixels(ragModule);
    exportAccumulate(ragModule);
    exportAccumulateFromBoundingBoxes(ragModule);
    #ifdef WITH_FASTFILTERS
    exportAccumulateFilters(ragModule);
    #endif
    exportEdgeCoordinates(ragModule);

    return ragModule.ptr();
}

