#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/malis/malis.hxx"

namespace py = pybind11;


namespace nifty {
namespace malis {

    template<unsigned DIM, typename DATA_TYPE, typename LABEL_TYPE>
    void exportMalisGradientT(py::module & malisModule){

        malisModule.def("malis_gradient",
           [](
                nifty::marray::PyView<DATA_TYPE, DIM+1> affinities,
                nifty::marray::PyView<LABEL_TYPE, DIM> groundtruth
           ){  
                typedef nifty::array::StaticArray<int64_t,DIM+1> Coord;
                Coord shape;
                for(int d = 0; d < DIM+1; ++d)
                    shape[d] = affinities.shape(d);
                nifty::marray::PyView<size_t, DIM+1> positiveGradients(shape.begin(), shape.end(), 0);
                nifty::marray::PyView<size_t, DIM+1> negativeGradients(shape.begin(), shape.end(), 0);
                {
                    py::gil_scoped_release allowThreads;
                    compute_malis_gradient<DIM>(affinities, groundtruth, positiveGradients, negativeGradients);
                }
                return std::make_tuple(positiveGradients, negativeGradients);
           },
           py::arg("affinities"),
           py::arg("groundtruth")
        );
    }

    void exportMalisGradient(py::module & malisModule){
        exportMalisGradientT<2,float,uint32_t>(malisModule);
        exportMalisGradientT<3,float,uint32_t>(malisModule);
    }
}
}
