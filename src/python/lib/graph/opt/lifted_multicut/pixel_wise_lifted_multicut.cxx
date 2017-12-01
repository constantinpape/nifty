#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nifty/graph/opt/lifted_multicut/fusion_move_based.hxx"
#include "nifty/graph/opt/lifted_multicut/pixel_wise.hxx"



#include <xtensor/xtensor.hpp>
#include <xtensor/xlayout.hpp>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace nifty{
namespace graph{
namespace opt{
namespace lifted_multicut{

    template<std::size_t DIM>
    void exportPixelWiseLmcStuffT(py::module & liftedMulticutModule) {


        typedef PixelWiseLmcObjective<DIM> ObjType;


        std::string objClsName = std::string("PixelWiseLmcObjective") + std::to_string(DIM) + std::string("D");
        py::class_<ObjType>(liftedMulticutModule, objClsName.c_str())


            .def(py::init(
            [](
                xt::pytensor<float,DIM+1> weights,
                xt::pytensor<int,  2> offsets
            ) { 
                return new ObjType(weights, offsets);
            }),
                py::arg("weights"),
                py::arg("offsets")
            )
            .def_property_readonly("shape",&ObjType::shape)
            .def("evaluate",
            [](
                const ObjType & self,
                const xt::pytensor<int, DIM> & labels
            ){
                return self.evaluate(labels);
            },
                py::arg("labels")
            )
            .def("optimize",
            [](
                const ObjType & self,
                typename ObjType::LmcFactoryBaseSharedPtr factory,
                xt::pytensor<uint64_t,  DIM> labels,
                const bool verbose
            ){
                return self.optimize(factory, labels, verbose);
            },
                py::arg("factory"),
                py::arg("labels"),
                py::arg("verbose") = false
            )
        ;


        typedef PixelWiseLmcConnetedComponentsFusion<DIM> CCFusionType;


        std::string fmClsName = std::string("PixelWiseLmcConnetedComponentsFusion") + std::to_string(DIM) + std::string("D");
        
        typedef typename CCFusionType::CCLmcFactoryBase CCLmcFactoryBase;
        typedef std::shared_ptr<CCLmcFactoryBase> CCLmcFactoryBaseSharedPtr;




        py::class_<CCFusionType>(liftedMulticutModule, fmClsName.c_str())
            


            .def(py::init(
            [](
                const ObjType & objective,
                CCLmcFactoryBaseSharedPtr solver_factory
            ) { 
                return new CCFusionType(objective,solver_factory);
            }),
                py::arg("objective"),
                py::arg("solver_factory"),
                py::keep_alive<1,2>()
            )

            .def("fuse",[](
                CCFusionType & self,
                xt::pytensor<uint64_t,  DIM> labels_a,
                xt::pytensor<uint64_t,  DIM> labels_b
            ){
                return self.fuse(labels_a, labels_b);
            })
            .def("fuse",[](
                CCFusionType & self,
                xt::pytensor<uint64_t,  DIM+1> labels
            ){
                return self.fuse(labels);
            })

        ;




    }

    void exportPixelWiseLmcStuff(py::module & liftedMulticutModule) {


        exportPixelWiseLmcStuffT<2>(liftedMulticutModule);
        exportPixelWiseLmcStuffT<3>(liftedMulticutModule);

        liftedMulticutModule.def("pixel_wise_lmc_edge_gt_2d",
            [](
                const xt::pytensor<uint64_t,2> gt,
                const xt::pytensor<int64_t, 2> offsets
            ){
                return nifty::graph::opt::lifted_multicut::pixel_wise_lmc_edge_gt_2d(gt, offsets);
            }
        );
            
    }



}
} // namespace nifty::graph::opt
}
}
