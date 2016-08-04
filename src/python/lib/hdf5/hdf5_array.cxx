#ifdef WITH_HDF5
#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"
#include "nifty/hdf5/hdf5_array.hxx"

namespace py = pybind11;



namespace nifty{
namespace hdf5{

    template<class T>
    void exportHdf5ArrayT(py::module & hdf5Module, const std::string & clsName) {
        typedef Hdf5Array<T> Hdf5ArrayType;
        py::class_<Hdf5ArrayType>(hdf5Module, clsName.c_str())

            .def(py::init<const hid_t & , const std::string &>())


            .def("__init__",[](
                Hdf5ArrayType & instance,
                const hid_t & groupHandle,
                const std::string & datasetName,
                std::vector<size_t> shape,
                std::vector<size_t> chunkShape
            ){
                NIFTY_CHECK_OP(shape.size(), == ,chunkShape.size(), 
                    "shape and chunk shape do not match");

                new (&instance) Hdf5ArrayType(groupHandle, datasetName,
                                              shape.begin(), shape.end(),
                                              chunkShape.begin());
            })

            .def_property_readonly("isChunked", &Hdf5ArrayType::isChunked)
            .def_property_readonly("ndim", &Hdf5ArrayType::dimension)
            .def_property_readonly("shape", [](const Hdf5ArrayType & array){
                return array.shape();
            })
            .def_property_readonly("chunkShape", [](const Hdf5ArrayType & array){
                return array.chunkShape();
            })
            .def("readSubarray",[](
                const Hdf5ArrayType & array,
                std::vector<size_t> roiBegin,
                std::vector<size_t> roiEnd
            ){
                const auto dim = array.dimension();
                NIFTY_CHECK_OP(roiBegin.size(),==,dim,"`roiBegin`has wrong size");
                NIFTY_CHECK_OP(roiEnd.size(),==,dim,  "`roiEnd`has wrong size");

                std::vector<size_t> shape(dim);
                for(size_t d=0; d<dim; ++d)
                    shape[d] = roiEnd[d] - roiBegin[d];
                
                nifty::marray::PyView<T> out(shape.begin(), shape.end());
                array.readSubarray(roiBegin.begin(), out);
                return out;
            })

            .def("writeSubarray",[](
                Hdf5ArrayType & array,
                std::vector<size_t> roiBegin,
                nifty::marray::PyView<T> in
            ){
                const auto dim = array.dimension();
                NIFTY_CHECK_OP(roiBegin.size(),==,dim,"`roiBegin`has wrong size");
                array.writeSubarray(roiBegin.begin(), in);
            })
        ;

    }


    void exportHdf5Array(py::module & hdf5Module) {

        exportHdf5ArrayT<uint8_t >(hdf5Module, "Hdf5ArrayUInt8");
        exportHdf5ArrayT<uint16_t>(hdf5Module, "Hdf5ArrayUInt16");
        exportHdf5ArrayT<uint32_t>(hdf5Module, "Hdf5ArrayUInt32");
        exportHdf5ArrayT<uint64_t>(hdf5Module, "Hdf5ArrayUInt64");

        exportHdf5ArrayT<int8_t >(hdf5Module, "Hdf5ArrayInt8");
        exportHdf5ArrayT<int16_t>(hdf5Module, "Hdf5ArrayInt16");
        exportHdf5ArrayT<int32_t>(hdf5Module, "Hdf5ArrayInt32");
        exportHdf5ArrayT<int64_t>(hdf5Module, "Hdf5ArrayInt64");

        exportHdf5ArrayT<float >(hdf5Module, "Hdf5ArrayFloat32");
        exportHdf5ArrayT<double >(hdf5Module, "Hdf5ArrayFloat64");
    }

}
}

#endif
