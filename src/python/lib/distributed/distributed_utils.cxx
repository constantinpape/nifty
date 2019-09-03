#ifdef WITH_Z5
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/distributed/distributed_utils.hxx"


namespace py = pybind11;

namespace nifty {
namespace distributed {

    void exportDistributedUtils(py::module & module) {

        module.def("serializeBlockMapping", [](const std::string & inputPath,
                                               const std::string & inputKey,
                                               const std::string & outputPath,
                                               const std::string & outputKey,
                                               const std::size_t numberOfLabels,
                                               const int numberOfThreads,
                                               const std::vector<std::size_t> & roiBegin,
                                               const std::vector<std::size_t> & roiEnd){
            py::gil_scoped_release allowThreads;
            serializeBlockMapping(inputPath, inputKey, outputPath, outputKey,
                                  numberOfLabels, numberOfThreads, roiBegin, roiEnd);
        }, py::arg("inputPath"), py::arg("inputKey"),
           py::arg("outputPath"), py::arg("outputKey"),
           py::arg("numberOfLabels"), py::arg("numberOfThreads"),
           py::arg("roiBegin")=std::vector<std::size_t>(), py::arg("roiEnd")=std::vector<std::size_t>());


        module.def("readBlockMapping", [](const std::string & path,
                                          const std::string & key,
                                          const std::vector<std::size_t> chunkId) {
            std::map<std::uint64_t, std::vector<std::array<int64_t, 6>>> mapping;
            readBlockMapping(path, key, chunkId, mapping);
            return mapping;
        }, py::arg("path"), py::arg("key"), py::arg("chunkId"));


        module.def("computeAndSerializeLabelOverlaps", [](const xt::pytensor<uint64_t, 3> & labels,
                                                          const xt::pytensor<uint64_t, 3> & values,
                                                          const std::string & path, const std::string & key,
                                                          const std::vector<std::size_t> & chunkId,
                                                          const bool withIgnoreLabel,
                                                          const uint64_t ignoreLabel){
            py::gil_scoped_release allowThreads;
            computeAndSerializeLabelOverlaps(labels, values, path, key, chunkId,
                                             withIgnoreLabel, ignoreLabel);
        }, py::arg("labels"), py::arg("values"),
           py::arg("path"), py::arg("key"), py::arg("chunkId"),
           py::arg("withIgnoreLabel")=false,
           py::arg("ignoreLabel")=0);


        module.def("mergeAndSerializeOverlaps", [](const std::string & inputPath,
                                                   const std::string & inputKey,
                                                   const std::string & outputPath,
                                                   const std::string & outputKey,
                                                   const bool max_overlap,
                                                   const int numberOfThreads,
                                                   const std::size_t labelBegin,
                                                   const std::size_t labelEnd,
                                                   const uint64_t ignoreLabel,
                                                   const bool serializeCount) {
            py::gil_scoped_release allowThreads;
            mergeAndSerializeOverlaps(inputPath, inputKey,
                                      outputPath, outputKey,
                                      max_overlap, numberOfThreads,
                                      labelBegin, labelEnd,
                                      ignoreLabel, serializeCount);
        }, py::arg("inputPath"), py::arg("inputKey"),
           py::arg("outputPath"), py::arg("outputKey"),
           py::arg("max_overlap"), py::arg("numberOfThreads"),
           py::arg("labelBegin"), py::arg("labelEnd"),
           py::arg("ignoreLabel")=0,
           py::arg("serializeCount")=false);


        module.def("computeLabelOverlaps", [](const xt::pytensor<uint64_t, 3> & labels,
                                              const xt::pytensor<uint64_t, 3> & gt){
            typedef std::unordered_map<uint64_t, std::size_t> OverlapType;
            std::unordered_map<uint64_t, OverlapType> overlaps;
            {
                py::gil_scoped_release allowThreads;
                computeLabelOverlaps(labels, gt, overlaps);
            }
            return overlaps;
        }, py::arg("labels"), py::arg("gt"));


        module.def("computeMaximumLabelOverlap", [](const xt::pytensor<uint64_t, 3> & labels,
                                                    const xt::pytensor<uint64_t, 3> & gt){
            typedef std::unordered_map<uint64_t, std::size_t> OverlapType;
            std::unordered_map<uint64_t, OverlapType> overlaps;
            {
                py::gil_scoped_release allowThreads;
                computeLabelOverlaps(labels, gt, overlaps);
            }
            const std::size_t n_nodes = overlaps.size();
            xt::pytensor<uint64_t, 1> max_overlaps = xt::zeros<uint64_t>({n_nodes});
            {
                py::gil_scoped_release allowThreads;
                for(const auto & node_ovlp: overlaps) {
                    const uint64_t node = node_ovlp.first;
                    const auto & ovlp = node_ovlp.second;

                    std::size_t max_ol = 0;
                    uint64_t max_label = 0;
                    for(const auto & elem: ovlp) {
                        if(elem.second > max_ol) {
                            max_ol = elem.second;
                            max_label = elem.first;
                        }
                    }
                    max_overlaps(node) = max_label;
                }
            }
            return max_overlaps;
        }, py::arg("labels"), py::arg("gt"));


        module.def("deserializeOverlapChunk", [](const std::string & path,
                                                 const std::string & key,
                                                 const std::vector<std::size_t> & chunkId) {
            typedef std::unordered_map<uint64_t, std::size_t> OverlapType;
            std::unordered_map<uint64_t, OverlapType> overlaps;
            const uint64_t maxLabelId = deserializeOverlapChunk(path, key, chunkId, overlaps);
            return std::make_pair(overlaps, maxLabelId);
        }, py::arg("path"), py::arg("key"), py::arg("chunkId"));


        module.def("serializeOverlapChunk", [](const std::string & path,
                                               const std::string & key,
                                               const std::vector<std::size_t> & chunkId,
                                               const std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::size_t>> & overlaps) {
            const z5::filesystem::handle::File file(path);
            auto ds = z5::openDataset(file, key);
            serializeLabelOverlaps(overlaps, std::move(ds), chunkId);
        }, py::arg("path"), py::arg("key"), py::arg("chunkId"), py::arg("overlap"));

    }
}
}
#endif
