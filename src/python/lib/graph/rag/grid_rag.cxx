#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nifty/python/converter.hxx"

#include "nifty/graph/rag/grid_rag.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/graph/rag/grid_rag_chunked.hxx"
#endif

#ifdef WITH_HDF5
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#endif

namespace py = pybind11;


namespace nifty{
namespace graph{



    using namespace py;
   
    template<class CLS>
    void removeFunctions(py::class_<CLS> & clsT){
        clsT
            .def("insertEdge", [](CLS * self,const uint64_t u,const uint64_t ){
                throw std::runtime_error("cannot insert edges into 'GridRag'");
            })
            .def("insertEdges",[](CLS * self, py::array_t<uint64_t> pyArray) {
                throw std::runtime_error("cannot insert edges into 'GridRag'");
            })
        ;
    }

     

    template<size_t DIM, class LABELS>
    void exportExpilictGridRagT(
        py::module & ragModule, 
        py::module & graphModule,
        const std::string & clsName,
        const std::string & facName
    ){
        py::object undirectedGraph = graphModule.attr("UndirectedGraph");
        typedef ExplicitLabelsGridRag<DIM, LABELS> GridRagType;

        auto clsT = py::class_<GridRagType>(ragModule, clsName.c_str(), undirectedGraph);
        removeFunctions<GridRagType>(clsT);

        ragModule.def(facName.c_str(),
            [](
               nifty::marray::PyView<LABELS, DIM> labels,
               const int numberOfThreads
            ){
                auto s = typename  GridRagType::Settings();
                s.numberOfThreads = numberOfThreads;
                ExplicitLabels<DIM, LABELS> explicitLabels(labels);
                auto ptr = new GridRagType(explicitLabels, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg_t< int >("numberOfThreads", -1 )
        );
    }

    #ifdef WITH_HDF5

    template<size_t DIM, class LABELS>
    void exportHdf5GridRagT(
        py::module & ragModule, 
        py::module & graphModule,
        const std::string & clsName,
        const std::string & facName
    ){
        py::object undirectedGraph = graphModule.attr("UndirectedGraph");
        
        typedef Hdf5Labels<DIM, LABELS> LabelsProxyType;
        typedef GridRag<DIM, LabelsProxyType >  GridRagType;


        const auto labelsProxyClsName = clsName + std::string("LabelsProxy");
        const auto labelsProxyFacName = facName + std::string("LabelsProxy");
        py::class_<LabelsProxyType>(ragModule, labelsProxyClsName.c_str())
            .def("hdf5Array",&LabelsProxyType::hdf5Array,py::return_value_policy::reference)
        ;

        ragModule.def(labelsProxyFacName.c_str(),
            [](
               const hdf5::Hdf5Array<LABELS> & hdf5Array,
               const int64_t numberOfLabels
            ){
                auto ptr = new LabelsProxyType(hdf5Array, numberOfLabels);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labels"),
            py::arg("numberOfLabels")
        );



        auto clsT = py::class_<GridRagType>(ragModule, clsName.c_str(), undirectedGraph);
        clsT
            .def("labelsProxy",&GridRagType::labelsProxy,py::return_value_policy::reference)
        ;

        removeFunctions<GridRagType>(clsT);





        ragModule.def(facName.c_str(),
            [](
                const LabelsProxyType & labelsProxy,
                std::vector<int64_t>  blockShape,
                const int numberOfThreads
            ){
                auto s = typename  GridRagType::Settings();
                s.numberOfThreads = numberOfThreads;

                if(blockShape.size() == DIM){
                    std::copy(blockShape.begin(), blockShape.end(), s.blockShape.begin());
                }
                else if(blockShape.size() == 1){
                    std::fill(s.blockShape.begin(), s.blockShape.end(), blockShape[0]);
                }
                else if(blockShape.size() != 0){
                    throw std::runtime_error("block shape has a non matching shape");
                }

                auto ptr = new GridRagType(labelsProxy, s);
                return ptr;
            },
            py::return_value_policy::take_ownership,
            py::keep_alive<0, 1>(),
            py::arg("labelsProxy"),
            py::arg_t< std::vector<int64_t>  >("blockShape", std::vector<int64_t>() ),
            py::arg_t< int >("numberOfThreads", -1 )
        );

    }

    #endif


    void exportGridRag(py::module & ragModule, py::module & graphModule) {

        exportExpilictGridRagT<2, uint32_t>(ragModule, graphModule, "ExplicitLabelsGridRag2D", "explicitLabelsGridRag2D");
        exportExpilictGridRagT<3, uint32_t>(ragModule, graphModule, "ExplicitLabelsGridRag3D", "explicitLabelsGridRag3D");
        
        #ifdef WITH_HDF5
        exportHdf5GridRagT<2, uint32_t>(ragModule, graphModule, "GridRagHdf5Labels2D", "gridRagHdf5");
        exportHdf5GridRagT<3, uint32_t>(ragModule, graphModule, "GridRagHdf5Labels3D", "gridRagHdf5");
        #endif


        // export ChunkedLabelsGridRagSliced
        #ifdef WITH_HDF5
        {
            py::object undirectedGraph = graphModule.attr("UndirectedGraph");
            typedef ChunkedLabelsGridRagSliced<uint32_t> ChunkedLabelsGridRagSliced;

            py::class_<ChunkedLabelsGridRagSliced>(ragModule, "ChunkedLabelsGridRagSliced", undirectedGraph)
                // remove a few methods
                .def("insertEdge", [](ChunkedLabelsGridRagSliced * self,const uint64_t u,const uint64_t ){
                    throw std::runtime_error("cannot insert edges into 'ChunkedLabelsGridRagSliced'");
                })
                .def("insertEdges",[](ChunkedLabelsGridRagSliced * self, py::array_t<uint64_t> pyArray) {
                    throw std::runtime_error("cannot insert edges into 'ChunkedLabelsGridRagSliced'");
                })
                .def("getTransitionEdge",&ChunkedLabelsGridRagSliced::getTransitionEdge)
                .def("isInnerSliceEdge",&ChunkedLabelsGridRagSliced::isInnerSliceEdge)
            ;
            
            // Thorstens changes
            //auto clsT = py::class_<ChunkedLabelsGridRagSliced>(ragModule, "ChunkedLabelsGridRagSliced", undirectedGraph);
            //removeFunctions<ExplicitLabelsGridRagType>(clsT);

            ragModule.def("chunkedLabelsGridRagSliced",
                [](const std::string & labelFile,
                   const std::string & labelKey,
                   const int numberOfThreads,
                   const bool forDeserialization
                ){
                    auto s = typename  ChunkedLabelsGridRagSliced::Settings();
                    s.numberOfThreads = numberOfThreads;
                    auto ptr = new ChunkedLabelsGridRagSliced(labelFile, labelKey, s);
                    return ptr;
                },
                py::return_value_policy::take_ownership,
                py::keep_alive<0, 1>(),
                py::arg("labelFile"),
                py::arg("labelKey"),
                py::arg_t< int >("numberOfThreads", -1 ),
                py::arg_t< bool >("forDeserialization", false)
            );
        }
        #endif
    }
        

} // end namespace graph
} // end namespace nifty
