#pragma once

#include <vector>
#include <functional>

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d.hxx"
#include "nifty/marray/marray.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/features/fastfilters_wrapper.hxx"
#include "vigra/accumulator.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"
#include "nifty/tools/array_tools.hxx"

#ifdef WITH_HDF5
#include "nifty/hdf5/hdf5_array.hxx"
#endif

namespace nifty{
namespace graph{

//
// helper functions for accumulate Edge Features
//

// calculate filters for given input with threadpool
template<class DATA_TYPE, class F, class COORD>
inline void calculateFilters(const marray::View<DATA_TYPE> & dataSqueezed,
        marray::View<float> & dataCopy,
        const COORD & sliceShape2,
        marray::View<float> & filter,
        parallel::ThreadPool & threadpool,
        const F & f) {

    typedef DATA_TYPE DataType;
    typedef COORD Coord;
    marray::View<float> dataView;
    if( typeid(DataType) == typeid(float) ) {
        dataView = dataSqueezed;
    }
    else {
        // copy the data (we don't use std::copy here, because iterators are terribly
        // slow for marrays)
        tools::forEachCoordinate(
            sliceShape2, [&dataCopy,&dataSqueezed](Coord coord){
                dataCopy(coord.asStdArray()) = (float) dataSqueezed(coord.asStdArray());
        });
        Coord base;
        Coord shape;
        for(int d = 0; d < dataCopy.dimension(); ++d) {
            base[d] = 0;
            shape[d] = dataCopy.shape(d);
        }
        dataView = dataCopy.view(base.begin(), shape.begin());
    }
    f(dataView, filter, threadpool);
}


// calculate filters for given input single threaded
template<class DATA_TYPE, class F, class COORD>
inline void calculateFilters(const marray::View<DATA_TYPE> & dataSqueezed,
        marray::View<float> & dataCopy,
        const COORD & sliceShape2,
        marray::View<float> & filter,
        const F & f,
        const bool preSmooth = false) {

    typedef DATA_TYPE DataType;
    typedef COORD Coord;
    marray::View<float> dataView;
    if( typeid(DataType) == typeid(float) ) {
        dataView = dataSqueezed;
    }
    else {
        // copy the data (we don't use std::copy here, because iterators are terribly
        // slow for marrays)
        tools::forEachCoordinate(
            sliceShape2, [&dataCopy,&dataSqueezed](Coord coord){
                dataCopy(coord.asStdArray()) = (float) dataSqueezed(coord.asStdArray());
        });
        Coord base;
        Coord shape;
        for(int d = 0; d < dataCopy.dimension(); ++d) {
            base[d] = 0;
            shape[d] = dataCopy.shape(d);
        }
        dataView = dataCopy.view(base.begin(), shape.begin());
    }
    f(dataView, filter, preSmooth);
}


template<class ACC_CHAIN_VECTOR, class HISTO_OPTS_VEC, class COORD, class LABEL_TYPE, class RAG>
inline void accumulateInnerSliceFeatures(ACC_CHAIN_VECTOR & channelAccChainVec,
        const HISTO_OPTS_VEC & histoOptionsVec,
        const COORD & sliceShape2,
        const marray::View<LABEL_TYPE> & labelsSqueezed,
        const int64_t sliceId,
        const int64_t inEdgeOffset,
        const RAG & rag,
        const marray::View<float> & filter
        ) {

    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
    typedef LABEL_TYPE LabelType;

    std::size_t pass = 1;
    std::size_t numberOfChannels = channelAccChainVec[0].size();

    // set minmax for accumulator chains
    for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge){
        for(int c = 0; c < numberOfChannels; ++c)
            channelAccChainVec[edge][c].setHistogramOptions(histoOptionsVec[c]);
    }

    // accumulate filter for the inner slice edges
    LabelType lU, lV;
    float fU, fV;
    VigraCoord vigraCoordU, vigraCoordV;
    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
        lU = labelsSqueezed(coord.asStdArray());
        for(int axis = 0; axis < 2; ++axis){
            Coord2 coord2 = coord;
            ++coord2[axis];
            if( coord2[axis] < sliceShape2[axis]) {
                lV = labelsSqueezed(coord2.asStdArray());
                if(lU != lV) {
                    vigraCoordU[0] = sliceId;
                    vigraCoordV[0] = sliceId;
                    for(int d = 1; d < 3; ++d){
                        vigraCoordU[d] = coord[d-1];
                        vigraCoordV[d] = coord2[d-1];
                    }
                    const auto edge = rag.findEdge(lU,lV) - inEdgeOffset;
                    for(int c = 0; c < numberOfChannels; ++c) {
                        fU = filter(c, coord[0], coord[1]);
                        fV = filter(c, coord2[0], coord2[1]);
                        channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
                        channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
                    }
                }
            }
        }
    });
}

// accumulate filter for the between slice edges
template<class ACC_CHAIN_VECTOR, class HISTO_OPTS_VEC, class COORD, class LABEL_TYPE, class RAG>
inline void accumulateBetweenSliceFeatures(ACC_CHAIN_VECTOR & channelAccChainVec,
        const HISTO_OPTS_VEC & histoOptionsVec,
        const COORD & sliceShape2,
        const marray::View<LABEL_TYPE> & labelsASqueezed,
        const marray::View<LABEL_TYPE> & labelsBSqueezed,
        const int64_t sliceIdA,
        const int64_t sliceIdB,
        const int64_t betweenEdgeOffset,
        const RAG & rag,
        const marray::View<float> & filterA,
        const marray::View<float> & filterB,
        const int zDirection
    ){

    typedef COORD Coord2;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;
    typedef LABEL_TYPE LabelType;

    std::size_t pass = 1;
    std::size_t numberOfChannels = channelAccChainVec[0].size();

    // set minmax for accumulator chains
    for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge){
        for(int c = 0; c < numberOfChannels; ++c)
            channelAccChainVec[edge][c].setHistogramOptions(histoOptionsVec[c]);
    }

    LabelType lU, lV;
    float fU, fV;
    VigraCoord vigraCoordU, vigraCoordV;
    nifty::tools::forEachCoordinate(sliceShape2, [&](const Coord2 coord){
        // labels are different for different slices by default!
        lU = labelsASqueezed(coord.asStdArray());
        lV = labelsBSqueezed(coord.asStdArray());
        vigraCoordU[0] = sliceIdA;
        vigraCoordV[0] = sliceIdB;
        for(int d = 1; d < 3; ++d){
            vigraCoordU[d] = coord[d-1];
            vigraCoordV[d] = coord[d-1];
        }
        const auto edge = rag.findEdge(lU,lV) - betweenEdgeOffset;
        if(zDirection==0) { // 0 -> take into account z and z + 1
            for(int c = 0; c < numberOfChannels; ++c) {
                fU = filterA(c, coord[0], coord[1]);
                fV = filterB(c, coord[0], coord[1]);
                channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
                channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
            }
        }
        else if(zDirection==1) { // 1 -> take into accout only z
            for(int c = 0; c < numberOfChannels; ++c) {
                fU = filterA(c, coord[0], coord[1]);
                channelAccChainVec[edge][c].updatePassN(fU, vigraCoordU, pass);
            }
        }
        else if(zDirection==2) { // 2 -> take into accout only z + 1
            for(int c = 0; c < numberOfChannels; ++c) {
                fV = filterB(c, coord[0], coord[1]);
                channelAccChainVec[edge][c].updatePassN(fV, vigraCoordV, pass);
            }
        }
    });
}


template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F_XY, class F_Z>
void accumulateEdgeFeaturesFromFiltersWithAccChain(
    const GridRagStacked2D<LABELS_PROXY> & rag,
    const DATA & data,
    const bool keepXYOnly,
    const bool keepZOnly,
    const parallel::ParallelOptions & pOpts,
    parallel::ThreadPool & threadpool,
    F_XY && fXY,
    F_Z && fZ,
    const int zDirection
){
    typedef LABELS_PROXY LabelsProxyType;
    typedef typename LabelsProxyType::LabelType LabelType;
    typedef typename DATA::DataType DataType;

    typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
    typedef tools::BlockStorage<DataType> DataBlockStorage;
    typedef tools::BlockStorage<float> FilterBlockStorage;

    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType>   AccChainVectorType;
    typedef std::vector<AccChainVectorType> ChannelAccChainVectorType;
    typedef typename features::ApplyFilters<2>::FiltersToSigmasType FiltersToSigmasType;

    const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    const auto & shape = rag.shape();
    const auto & labelsProxy = rag.labelsProxy();

    // sigmas and filters to sigmas: TODO make accessible
    std::vector<double> sigmas({1.6,4.2,8.2});
    FiltersToSigmasType filtersToSigmas({ { true, true, true},      // GaussianSmoothing
                                          { true, true, true},      // LaplacianOfGaussian
                                          { false, false, false},   // GaussianGradientMagnitude
                                          { true, true, true } });  // HessianOfGaussianEigenvalues

    features::ApplyFilters<2> applyFilters(sigmas, filtersToSigmas);
    std::size_t numberOfChannels = applyFilters.numberOfChannels();

    uint64_t numberOfSlices = shape[0];

    Coord2 sliceShape2({shape[1], shape[2]});
    Coord sliceShape3({1LL, shape[1], shape[2]});
    Coord filterShape({int64_t(numberOfChannels), shape[1], shape[2]});

    // filter computation and accumulation
    // FIXME we only support 1 pass for now
    //for(auto pass = 1; pass <= channelAccChainVector.front().front().passesRequired(); ++pass) {
    //int pass = 1;
    {
        // accumulate inner slice feature

        // TODO we could do this less memory consuming, if we allocate less data here and
        // allocate some in every thread
        LabelBlockStorage  labelsAStorage(threadpool, sliceShape3, actualNumberOfThreads);
        LabelBlockStorage  labelsBStorage(threadpool, sliceShape3, actualNumberOfThreads);
        FilterBlockStorage filterAStorage(threadpool, filterShape, actualNumberOfThreads);
        FilterBlockStorage filterBStorage(threadpool, filterShape, actualNumberOfThreads);
        // we only need one data storage
        DataBlockStorage   dataStorage(threadpool, sliceShape3, actualNumberOfThreads);
        // storage for the data we have to copy if type of data is not float
        //FilterBlockStorage dataCopyStorage(threadpool, sliceShape2, actualNumberOfThreads);

        // process slice 0 to find min and max for histogram opts
        Coord begin0({0LL, 0LL, 0LL});
        Coord end0(  {1LL, shape[1], shape[2]});

        auto data0 = dataStorage.getView(0);
        tools::readSubarray(data, begin0, end0, data0);
        auto data0Squeezed = data0.squeezedView();

        //auto dataCopy = dataCopyStorage.getView(0); // in case we need to copy data for non-float type
        marray::Marray<float> dataCopy;
        if( typeid(DataType) != typeid(float) )
            dataCopy.resize(sliceShape2.begin(), sliceShape2.end());
        auto filter0 = filterAStorage.getView(0);

        // apply filters in parallel
        calculateFilters(data0Squeezed,
                dataCopy,
                sliceShape2,
                filter0,
                threadpool,
                applyFilters);

        // apply filters with pre smoothing
        // TODO benchmark this!
        //calculateFilters(data0Squeezed,
        //        dataCopy,
        //        sliceShape2,
        //        filter0,
        //        applyFilters,
        //        true);

        std::vector<vigra::HistogramOptions> histoOptionsVec(numberOfChannels);
        Coord cShape({1LL,sliceShape2[0],sliceShape2[1]});
        parallel::parallel_foreach(threadpool, numberOfChannels, [&](const int tid, const int64_t c){
            auto & histoOpts = histoOptionsVec[c];
            Coord cBegin({c,0LL,0LL});
            auto channelView = filter0.view(cBegin.begin(), cShape.begin());
            auto minMax = std::minmax_element(channelView.begin(), channelView.end());
            auto min = *(minMax.first);
            auto max = *(minMax.second);
            histoOpts.setMinMax(min,max);
        });

        // construct slice pairs for processing in parallel
        std::vector<std::pair<int64_t,int64_t>> slicePairs;
        int64_t lowerSliceId = 0;
        int64_t upperSliceId = 1;
        while(upperSliceId < numberOfSlices) {
            slicePairs.emplace_back(std::make_pair(lowerSliceId,upperSliceId));
            ++lowerSliceId;
            ++upperSliceId;
        }

        parallel::parallel_foreach(threadpool, slicePairs.size(), [&](const int tid, const int64_t pairId){

            std::cout << "Processing slice pair: " << pairId << " / " << slicePairs.size() << std::endl;
            int64_t sliceIdA = slicePairs[pairId].first; // lower slice
            int64_t sliceIdB = slicePairs[pairId].second;// upper slice

            // compute the filters for slice A
            Coord beginA ({sliceIdA, 0LL, 0LL});
            Coord endA({sliceIdA+1, shape[1], shape[2]});

            auto labelsA = labelsAStorage.getView(tid);
            labelsProxy.readSubarray(beginA, endA, labelsA);
            auto labelsASqueezed = labelsA.squeezedView();

            auto dataA = dataStorage.getView(tid);
            tools::readSubarray(data, beginA, endA, dataA);
            auto dataASqueezed = dataA.squeezedView();

            //auto dataCopy = dataCopyStorage.getView(tid);
            marray::Marray<float> dataCopy;
            if( typeid(DataType) != typeid(float) )
                dataCopy.resize(sliceShape2.begin(), sliceShape2.end());

            auto filterA = filterAStorage.getView(tid);
            calculateFilters(dataASqueezed,
                    dataCopy,
                    sliceShape2,
                    filterA,
                    applyFilters,
                    true); // presmoothing

            // acccumulate the inner slice features
            // only if not keepZOnly and if we have at least one edge in this slice
            // (no edge can happend for defected slices)
            if( rag.numberOfInSliceEdges(sliceIdA) > 0 && !keepZOnly) {
                auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdA);
                // make new acc chain vector
                ChannelAccChainVectorType channelAccChainVec(
                        rag.numberOfInSliceEdges(sliceIdA),
                        AccChainVectorType(numberOfChannels)
                );
                accumulateInnerSliceFeatures(channelAccChainVec,
                        histoOptionsVec,
                        sliceShape2,
                        labelsASqueezed,
                        sliceIdA,
                        inEdgeOffset,
                        rag,
                        filterA);
                fXY(channelAccChainVec, inEdgeOffset);
            }

            // process upper slice
            Coord beginB = Coord({sliceIdB,   0LL,       0LL});
            Coord endB   = Coord({sliceIdB+1, shape[1], shape[2]});
            auto filterB = filterBStorage.getView(tid);
            marray::View<LabelType> labelsBSqueezed;

            // read labels, data and calculate the filters for upper slice
            // do if we are not keeping only xy edges or
            // if we are at the last slice (which is never a lower slice and
            // must hence be accumulated extra)
            if(!keepXYOnly || sliceIdB == numberOfSlices - 1 ) {
                // read labels
                auto labelsB = labelsBStorage.getView(tid);
                labelsProxy.readSubarray(beginB, endB, labelsB);
                labelsBSqueezed = labelsB.squeezedView();
                // read data
                auto dataB = dataStorage.getView(tid);
                tools::readSubarray(data, beginB, endB, dataB);
                auto dataBSqueezed = dataB.squeezedView();
                // calc filter
                calculateFilters(dataBSqueezed,
                        dataCopy,
                        sliceShape2,
                        filterB,
                        applyFilters,
                        true); // activate pre-smoothing
            }

            // acccumulate the between slice features
            if(!keepXYOnly) {
                auto betweenEdgeOffset = rag.betweenSliceEdgeOffset(sliceIdA);
                auto accOffset = rag.betweenSliceEdgeOffset(sliceIdA) - rag.numberOfInSliceEdges();
                // make new acc chain vector
                ChannelAccChainVectorType channelAccChainVec(
                    rag.numberOfInBetweenSliceEdges(sliceIdA),
                    AccChainVectorType(numberOfChannels)
                );
                // accumulate features for the in between slice edges
                accumulateBetweenSliceFeatures(channelAccChainVec,
                        histoOptionsVec,
                        sliceShape2,
                        labelsASqueezed,
                        labelsBSqueezed,
                        sliceIdA,
                        sliceIdB,
                        betweenEdgeOffset,
                        rag,
                        filterA,
                        filterB,
                        zDirection);
                fZ(channelAccChainVec, accOffset);
            }

            // accumulate the inner slice features for the last slice, which is never a lower slice
            if(!keepZOnly && (sliceIdB == numberOfSlices - 1 && rag.numberOfInSliceEdges(sliceIdB) > 0)) {
                auto inEdgeOffset = rag.inSliceEdgeOffset(sliceIdB);
                // make new acc chain vector
                ChannelAccChainVectorType channelAccChainVec(
                    rag.numberOfInSliceEdges(sliceIdB),
                    AccChainVectorType(numberOfChannels)
                );
                accumulateInnerSliceFeatures(channelAccChainVec,
                        histoOptionsVec,
                        sliceShape2,
                        labelsBSqueezed,
                        sliceIdB,
                        inEdgeOffset,
                        rag,
                        filterB);
                fXY(channelAccChainVec, inEdgeOffset);
            }

        });
    }
    std::cout << "Slices done" << std::endl;
}


// 9 features per channel
template<class LABELS_PROXY, class DATA, class OUTPUT>
void accumulateEdgeFeaturesFromFilters(
    const GridRagStacked2D<LABELS_PROXY> & rag,
    const DATA & data,
    OUTPUT & edgeFeaturesOutXY,
    OUTPUT & edgeFeaturesOutZ,
    const bool keepXYOnly,
    const bool keepZOnly,
    const int zDirection = 0,
    const int numberOfThreads = -1
){
    namespace acc = vigra::acc;
    typedef float DataType;

    typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

    typedef acc::Select<
        acc::DataArg<1>,
        acc::Mean,        //1
        acc::Variance,    //1
        Quantiles         //7
    > SelectType;
    typedef acc::StandAloneAccumulatorChain<3, DataType, SelectType> AccChainType;

    // threadpool
    nifty::parallel::ParallelOptions pOpts(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpts);

    // general accumulator function
    auto accFunction = [&threadpool](
        const std::vector<std::vector<AccChainType>> & channelAccChainVec,
        const uint64_t edgeOffset,
        OUTPUT & edgeFeaturesOut
    ){
        using namespace vigra::acc;
        typedef array::StaticArray<int64_t, 2> FeatCoord;

        const auto nStats = 9;
        const auto nEdges    = channelAccChainVec.size();
        const auto nChannels = channelAccChainVec.front().size();

        marray::Marray<DataType> featuresTemp({nEdges,nChannels*nStats});
        for(int64_t edge = 0; edge < channelAccChainVec.size(); ++edge) {
            const auto & edgeAccChainVec = channelAccChainVec[edge];
            auto cOffset = 0;
            vigra::TinyVector<float,7> quantiles;
            for(int c = 0; c < nChannels; ++c) {
                const auto & chain = edgeAccChainVec[c];
                const auto mean = get<acc::Mean>(chain);
                featuresTemp(edge, cOffset) = replaceIfNotFinite(mean,0.0);
                featuresTemp(edge, cOffset+1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                quantiles = get<Quantiles>(chain);
                for(auto qi=0; qi<7; ++qi)
                    featuresTemp(edge, cOffset+2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                cOffset += nStats;
            }
        }

        FeatCoord begin({int64_t(edgeOffset),0LL});
        FeatCoord end({edgeOffset+nEdges,uint64_t(nChannels*nStats)});

        tools::writeSubarray(edgeFeaturesOut, begin, end, featuresTemp);
    };

    // instantiation of accumulators for xy / z edges
    auto accFunctionXY = std::bind(accFunction,
            std::placeholders::_1,
            std::placeholders::_2,
            std::ref(edgeFeaturesOutXY));
    auto accFunctionZ = std::bind(accFunction,
            std::placeholders::_1,
            std::placeholders::_2,
            std::ref(edgeFeaturesOutZ));

    accumulateEdgeFeaturesFromFiltersWithAccChain<AccChainType>(
        rag,
        data,
        keepXYOnly,
        keepZOnly,
        pOpts,
        threadpool,
        accFunctionXY,
        accFunctionZ,
        zDirection
        );
}


// TODO use the proper helper functions here !
template<class EDGE_ACC_CHAIN, class LABELS_PROXY, class DATA, class F>
void accumulateSkipEdgeFeaturesFromFiltersWithAccChain(
    const GridRagStacked2D<LABELS_PROXY> & rag,
    const DATA & data,
    const std::vector<std::pair<uint64_t,uint64_t>> & skipEdges,
    const std::vector<std::size_t> & skipRanges,
    const std::vector<std::size_t> & skipStarts,
    const parallel::ParallelOptions & pOpts,
    parallel::ThreadPool & threadpool,
    F && f,
    const int zDirection
){
    typedef std::pair<uint64_t,uint64_t> SkipEdgeStorage;

    typedef LABELS_PROXY LabelsProxyType;
    typedef typename DATA::DataType DataType;
    typedef typename vigra::MultiArrayShape<3>::type VigraCoord;

    typedef typename LabelsProxyType::BlockStorageType LabelBlockStorage;
    typedef tools::BlockStorage<DataType> DataBlockStorage;
    typedef tools::BlockStorage<float> FilterBlockStorage;

    typedef array::StaticArray<int64_t, 3> Coord;
    typedef array::StaticArray<int64_t, 2> Coord2;

    typedef EDGE_ACC_CHAIN EdgeAccChainType;
    typedef std::vector<EdgeAccChainType>   AccChainVectorType;     // holds acc chains over the filter channels for each edge
    typedef std::vector<AccChainVectorType> EdgeAccChainVectorType; // holds acc chains over the edges

    typedef typename features::ApplyFilters<2>::FiltersToSigmasType FiltersToSigmasType;

    const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

    const auto & shape = rag.shape();
    const auto & labelsProxy = rag.labelsProxy();

    // sigmas and filters to sigmas: TODO make accessible
    std::vector<double> sigmas({1.6,4.2,8.2});
    FiltersToSigmasType filtersToSigmas({ { true, true, true},      // GaussianSmoothing
                                          { true, true, true},      // LaplacianOfGaussian
                                          { false, false, false},   // GaussianGradientMagnitude
                                          { true, true, true } });  // HessianOfGaussianEigenvalues

    features::ApplyFilters<2> applyFilters(sigmas, filtersToSigmas);
    std::size_t numberOfChannels = applyFilters.numberOfChannels();

    Coord2 sliceShape2({shape[1], shape[2]});
    Coord sliceShape3({1LL,shape[1], shape[2]});
    Coord filterShape({int64_t(numberOfChannels), shape[1], shape[2]});

    // filter computation and accumulation
    // FIXME we only support 1 pass for now
    //for(auto pass = 1; pass <= channelAccChainVector.front().front().passesRequired(); ++pass) {
    int pass = 1;
    {
        LabelBlockStorage labelsAStorage(threadpool, sliceShape3, 1);
        LabelBlockStorage labelsBStorage(threadpool, sliceShape3, 1);
        FilterBlockStorage filterAStorage(threadpool, filterShape, 1);
        FilterBlockStorage filterBStorage(threadpool, filterShape, 1);
        // only need one data storage
        DataBlockStorage dataStorage(threadpool, sliceShape3, 1);
        // storage for the data we need to copy for uint8 input
        FilterBlockStorage dataCopyStorage(threadpool, sliceShape2, 1);

        // get unique lower slices with skip edges
        std::vector<std::size_t> lowerSlices;
        tools::uniques(skipStarts, lowerSlices);
        auto lowest = int64_t(lowerSlices[0]);

        // get upper slices with skip edges for each lower slice and number of skip edges for each lower slice
        std::map<std::size_t,std::vector<std::size_t>> skipSlices;
        std::map<std::size_t,size_t> numberOfSkipEdgesPerSlice;

        // initialize the maps
        std::cout << "Lower slices with skip edges:" << std::endl;
        for(auto sliceId : lowerSlices) {
            std::cout << sliceId << std::endl;
            skipSlices[sliceId] = std::vector<std::size_t>();
            numberOfSkipEdgesPerSlice[sliceId] = 0;
        }

        // determine the number of skip edges starting from each slice
        // and fill the conversion of uv-ids to edge-id
        std::map<SkipEdgeStorage, std::size_t> skipUvsToIds;
        for(std::size_t skipId = 0; skipId < skipEdges.size(); ++skipId) {

            // find the source slice of this skip edge and increment the number of edges in the slice
            auto sliceId = skipStarts[skipId];
            ++numberOfSkipEdgesPerSlice[sliceId];

            // add the target slice of this skip edge to the target slices of the source slice
            auto targetSlice = sliceId + skipRanges[skipId];
            auto & thisSkipSlices = skipSlices[sliceId];
            if(std::find(thisSkipSlices.begin(), thisSkipSlices.end(), targetSlice) == thisSkipSlices.end() )
                thisSkipSlices.push_back(targetSlice);

            // fill the uvs to ids map
            skipUvsToIds[skipEdges[skipId]] = skipId;
        }

        std::vector<vigra::HistogramOptions> histoOptionsVec(numberOfChannels);

        std::size_t skipEdgeOffset = 0;
        int countSlice = 0;
        for(auto sliceId : lowerSlices) {

            std::cout << countSlice++ << " / " << lowerSlices.size() << std::endl;
            std::cout << "Finding features for skip edges from slice " << sliceId << std::endl;

            Coord beginA({int64_t(sliceId),0LL,0LL});
            Coord endA(  {int64_t(sliceId+1),shape[1],shape[2]});
            auto labelsA = labelsAStorage.getView(0);
            labelsProxy.readSubarray(beginA, endA, labelsA);
            auto labelsASqueezed = labelsA.squeezedView();

            auto dataA = dataStorage.getView(0);
            tools::readSubarray(data, beginA, endA, dataA);
            auto dataASqueezed = dataA.squeezedView();

            auto dataCopy = dataCopyStorage.getView(0); // in case we need to copy data for non-float type
            auto filterA = filterAStorage.getView(0);

            // apply filters in parallel
            calculateFilters(dataASqueezed,
                    dataCopy,
                    sliceShape2,
                    filterA,
                    threadpool,
                    applyFilters);

            // set the correct histogram for each filter from lowest slice
            if(sliceId == lowest){
                Coord cShape({1LL,sliceShape2[0],sliceShape2[1]});
                parallel::parallel_foreach(threadpool, numberOfChannels, [&](const int tid, const int64_t c){
                    auto & histoOpts = histoOptionsVec[c];

                    Coord cBegin({c,0LL,0LL});
                    auto channelView = filterA.view(cBegin.begin(), cShape.begin());
                    auto minMax = std::minmax_element(channelView.begin(), channelView.end());
                    auto min = *(minMax.first);
                    auto max = *(minMax.second);
                    histoOpts.setMinMax(min,max);
                });
            }

            auto numberOfSkipEdgesInSlice = numberOfSkipEdgesPerSlice[sliceId];

            // init the edge acc vectors for all threads wth the number of skip edges in this slice
            std::vector<EdgeAccChainVectorType> perThreadAccChainVector(actualNumberOfThreads);  // vector<vector<vector<AccChain>>>
            for(int t = 0; t < actualNumberOfThreads; ++t) {                                     // thread edge   channel
                auto & accChainVector = perThreadAccChainVector[t];
                accChainVector.resize(numberOfSkipEdgesInSlice);

                // set the histo opts for each edge acc chain vector and channel
                parallel::parallel_foreach(threadpool, numberOfSkipEdgesInSlice, [&](const int tid, const int64_t skipEdge){
                        auto & accChainVec = accChainVector[skipEdge];
                        accChainVec.resize(numberOfChannels);
                        for(int c = 0; c < numberOfChannels; ++c){
                            accChainVec[c].setHistogramOptions(histoOptionsVec[c]);
                        }
                });
            }

            Coord beginB;
            Coord endB;
            // iterate over all upper slices that have skip edges with this slice
            for(auto nextId : skipSlices[sliceId] ) {
                std::cout << "to slice " << nextId << std::endl;

                beginB = Coord({int64_t(nextId),0LL,0LL});
                endB   = Coord({int64_t(nextId+1),shape[1],shape[2]});

                auto labelsB = labelsBStorage.getView(0);
                labelsProxy.readSubarray(beginB, endB, labelsB);
                auto labelsBSqueezed = labelsB.squeezedView();

                auto dataB = dataStorage.getView(0);
                tools::readSubarray(data, beginB, endB, dataB);
                auto dataBSqueezed = dataB.squeezedView();

                auto dataCopy = dataCopyStorage.getView(0); // in case we need to copy data for non-float type
                auto filterB = filterBStorage.getView(0);

                // apply filters in parallel
                calculateFilters(dataBSqueezed,
                        dataCopy,
                        sliceShape2,
                        filterB,
                        threadpool,
                        applyFilters);

                // accumulate filter for the between slice edges
                nifty::tools::parallelForEachCoordinate(threadpool, sliceShape2, [&](const int tid, const Coord2 coord){

                    auto & threadData = perThreadAccChainVector[tid];

                    // labels are different for different slices by default!
                    const auto lU = labelsASqueezed(coord.asStdArray());
                    const auto lV = labelsBSqueezed(coord.asStdArray());

                    // we search for the skipId corresponding to this pair of uv-ids
                    // for slices with partial defects, not all uv-pairs correspond to a skip-edge.
                    // in that case, we simply continue without doing anything
                    auto skipPair = std::make_pair(static_cast<uint64_t>(lU), static_cast<uint64_t>(lV));
                    auto mapIterator = skipUvsToIds.find(skipPair);
                    if(mapIterator == skipUvsToIds.end()) {
                        return;
                    }

                    // the in slice id corresponds to the global id - the offset corresponding to
                    // the current source slice
                    auto skipId = mapIterator->second - skipEdgeOffset;

                    auto & accChainVec = threadData[skipId];

                    VigraCoord vigraCoordU;
                    VigraCoord vigraCoordV;
                    vigraCoordU[0] = sliceId;
                    vigraCoordV[0] = nextId;
                    for(int d = 1; d < 3; ++d){
                        vigraCoordU[d] = coord[d-1];
                        vigraCoordV[d] = coord[d-1];
                    }

                    if(zDirection==0) { // 0 -> take into account z and z + 1
                        for(int c = 0; c < numberOfChannels; ++c) {
                            const auto fU = filterA(c, coord[0], coord[1]);
                            const auto fV = filterB(c, coord[0], coord[1]);
                            accChainVec[c].updatePassN(fU, vigraCoordU, pass);
                            accChainVec[c].updatePassN(fV, vigraCoordV, pass);
                        }
                    }
                    else if(zDirection==1) { // 1 -> take into accout only z
                        for(int c = 0; c < numberOfChannels; ++c) {
                            const auto fU = filterA(c, coord[0], coord[1]);
                            accChainVec[c].updatePassN(fU, vigraCoordU, pass);
                        }
                    }
                    else if(zDirection==2) { // 2 -> take into accout only z + 1
                        for(int c = 0; c < numberOfChannels; ++c) {
                            const auto fV = filterB(c, coord[0], coord[1]);
                            accChainVec[c].updatePassN(fV, vigraCoordV, pass);
                        }
                    }
                });
            }

            // we merge the accumulator chains into the zeros acc chain in parallel over the skip edges
            auto & accumulatorVectorDest = perThreadAccChainVector[0];
            parallel::parallel_foreach(threadpool, numberOfSkipEdgesInSlice, [&](const int tid, const int64_t skipId){

                auto & accVecDest = accumulatorVectorDest[skipId];

                for(int t = 1; t < actualNumberOfThreads; ++t) {
                    auto & accVecSrc = perThreadAccChainVector[t][skipId];
                     for(int c = 0; c < numberOfChannels; ++c) {
                        accVecDest[c].merge(accVecSrc[c]);
                    }
                }
            });

            f(accumulatorVectorDest, skipEdgeOffset);
            skipEdgeOffset += numberOfSkipEdgesInSlice;
        }
    }
}


// 9 features per channel
template<class LABELS_PROXY, class DATA, class OUTPUT>
void accumulateSkipEdgeFeaturesFromFilters(
    const GridRagStacked2D<LABELS_PROXY> & rag,
    const DATA & data,
    OUTPUT & edgeFeaturesOut,
    const std::vector<std::pair<uint64_t,uint64_t>> & skipEdges,
    const std::vector<std::size_t> & skipRanges,
    const std::vector<std::size_t> & skipStarts,
    const int zDirection = 0,
    const int numberOfThreads = -1
){
    namespace acc = vigra::acc;
    typedef float DataType;

    typedef acc::UserRangeHistogram<40>            SomeHistogram;   //binCount set at compile time
    typedef acc::StandardQuantiles<SomeHistogram > Quantiles;

    typedef acc::Select<
        acc::DataArg<1>,
        acc::Mean,        //1
        acc::Variance,    //1
        Quantiles         //7
    > SelectType;
    typedef acc::StandAloneAccumulatorChain<3, DataType, SelectType> AccChainType;

    // threadpool
    nifty::parallel::ParallelOptions pOpts(numberOfThreads);
    nifty::parallel::ThreadPool threadpool(pOpts);

    const auto nStats = 9;

    accumulateSkipEdgeFeaturesFromFiltersWithAccChain<AccChainType>(
        rag,
        data,
        skipEdges,
        skipRanges,
        skipStarts,
        pOpts,
        threadpool,
        [&](
            const std::vector<std::vector<AccChainType>> & channelAccChainVec,
            const uint64_t edgeOffset
        ){
            using namespace vigra::acc;
            typedef array::StaticArray<int64_t, 2> FeatCoord;

            const auto nEdges    = channelAccChainVec.size();
            const auto nChannels = channelAccChainVec.front().size();

            marray::Marray<DataType> featuresTemp({nEdges,nChannels*nStats});

            parallel::parallel_foreach(threadpool, channelAccChainVec.size(),
                [&](const int tid, const int64_t edge){

                const auto & edgeAccChainVec = channelAccChainVec[edge];
                auto cOffset = 0;
                vigra::TinyVector<float,7> quantiles;
                for(int c = 0; c < nChannels; ++c) {
                    const auto & chain = edgeAccChainVec[c];
                    const auto mean = get<acc::Mean>(chain);
                    featuresTemp(edge, cOffset) = replaceIfNotFinite(mean,0.0);
                    featuresTemp(edge, cOffset+1) = replaceIfNotFinite(get<acc::Variance>(chain), 0.0);
                    quantiles = get<Quantiles>(chain);
                    for(auto qi=0; qi<7; ++qi)
                        featuresTemp(edge, cOffset+2+qi) = replaceIfNotFinite(quantiles[qi], mean);
                    cOffset += nStats;
                }
            });

            FeatCoord begin({int64_t(edgeOffset),0LL});
            FeatCoord end({edgeOffset+nEdges,uint64_t(nChannels*nStats)});

            tools::writeSubarray(edgeFeaturesOut, begin, end, featuresTemp);
        },
        zDirection
    );
}


} // namespace graph
} // namespace nifty
