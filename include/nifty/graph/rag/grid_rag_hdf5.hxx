#pragma once
#ifndef NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX
#define NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#include "nifty/graph/rag/detail_rag/compute_grid_rag_hdf5.hxx"


namespace nifty{
namespace graph{

template<unsigned int DIM, class LABEL_TYPE>
using Hdf5LabelsGridRag = GridRag<DIM, Hdf5Labels<DIM, LABEL_TYPE> >; 


} // end namespace graph
} // end namespace nifty

#endif /* NIFTY_GRAPH_RAG_GRID_RAG_HDF5_HXX */
