#pragma once
#include <vector>
#include <unordered_map>

#include "nifty/xtensor/xtensor.hxx"
#include "nifty/array/arithmetic_array.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace ground_truth{

    template<class LABEL_TYPE = uint64_t, class COUNT_TYPE = uint64_t>
    class Overlap{
    public:

        typedef LABEL_TYPE LabelType;
        typedef COUNT_TYPE CountType;
        typedef std::unordered_map<LabelType, CountType> MapType;

        template<class SET_A_ITER, class SET_B_ITER>
        Overlap(
            const uint64_t maxLabelSetA,
            SET_A_ITER aBegin,
            SET_A_ITER aEnd,
            SET_B_ITER bBegin
        )
        :   overlaps_(maxLabelSetA+1),
            counts_(maxLabelSetA+1){

            fill(aBegin, aEnd, bBegin);
        }

        template<class LABELS_A, class LABELS_B>
        Overlap(
            const uint64_t maxLabelSetA,
            const LABELS_A & arrayA,
            const LABELS_B & arrayB
        )
        :   overlaps_(maxLabelSetA+1),
            counts_(maxLabelSetA+1){

            const auto dimA = arrayA.dimension();
            const auto dimB = arrayB.dimension();
            NIFTY_CHECK_OP(dimA,==,dimB,"dimension mismatch in Overlap::Overlap")

            for(auto d=0; d<dimA; ++d){
                NIFTY_CHECK_OP(arrayA.shape()[d],==,arrayB.shape()[d],"shape mismatch in Overlap::Overlap")
            }

            if(dimA == 1){
                fill<1>(arrayA, arrayB);
            }
            else if(dimA == 2){
                fill<2>(arrayA, arrayB);
            }
            else if(dimA == 3){
                fill<3>(arrayA, arrayB);
            }
            else if(dimA == 4){
                fill<4>(arrayA, arrayB);
            }
            else{
                auto aBegin = arrayA.begin();
                auto aEnd = arrayA.end();
                auto bBegin = arrayB.begin();
                fill(aBegin, aEnd, bBegin);
            }

        }


        double differentOverlap(const LabelType u, const LabelType v)const{

            const auto & olU = overlaps_[u];
            const auto & olV = overlaps_[v];
            const auto sU = float(counts_[u]);
            const auto sV = float(counts_[v]);
            auto isDiff = 0.0;
            for(const auto & keyAndSizeU : olU)
            for(const auto & keyAndSizeV : olV){

                auto keyU =  keyAndSizeU.first;
                auto rSizeU = float(keyAndSizeU.second)/sU;
                auto keyV =  keyAndSizeV.first;
                auto rSizeV = float(keyAndSizeV.second)/sV;

                if(keyU != keyV){
                    isDiff += (rSizeU * rSizeV);
                }
            }
            return isDiff;
        }

        double bleeding(const LabelType u)const{
            const COUNT_TYPE size = counts_[u];
            const auto & ol = overlaps_[u];

            std::vector<COUNT_TYPE> olCount;
            olCount.reserve(ol.size());

            COUNT_TYPE maxOlCount = 0;

            for(const auto & kv : ol){
                maxOlCount = std::max(maxOlCount, kv.second);
            }
            return 1.0 - (double(size) - double(maxOlCount))/size;

        }
        const std::vector<CountType> & counts()const{
            return counts_;
        };
        const std::vector<MapType> & overlaps()const{
            return overlaps_;
        };


        LabelType maxOverlappingLabel(const LabelType u )const{
            const auto & ol = overlaps_[u];
            CountType maxOl = 0;
            LabelType maxL = 0 ;
            for(const auto & kv : ol){
                if(kv.second > maxOl){
                    maxOl = kv.second;
                    maxL = kv.first;
                }
            }
            return maxL;
        }
        /**
         * @brief      find the maximum overlapping label and ignore zeros,
         *             except if zero is the only overlap.
         *
         * @param[in]  u     query label
         *
         * @return     maximum overlapping label
         */
        LabelType maxOverlappingLabelDownvoteZeros(const LabelType u )const{
            const auto & ol = overlaps_[u];
            CountType maxOl = 0;
            LabelType maxL = 0;
            const auto size = ol.size();

            for(const auto & kv : ol){
                if(size==0){
                    return kv.first;
                }
                if(kv.first!=LabelType(0) && kv.second > maxOl){
                    maxOl = kv.second;
                    maxL = kv.first;
                }
            }
            return maxL;
        }
        std::pair<LabelType,bool> maxOverlappingNonZeroLabel(const LabelType u )const{
            const auto & ol = overlaps_[u];
            bool found = false;
            CountType maxOl = 0;
            LabelType maxL = 0;
            for(const auto & kv : ol){
                if(kv.first!=LabelType(0) && kv.second > maxOl){
                    maxOl = kv.second;
                    maxL = kv.first;
                    found = true;
                }
            }
            return std::pair<LabelType,bool>(maxL, found);
        }

        bool isOverlappingWithZero(const LabelType u )const{
            const auto & ol = overlaps_[u];
            return ol.find(LabelType(0)) != ol.end();
        }



    private:

        template<class SET_A_ITER, class SET_B_ITER>
        void fill(
            SET_A_ITER aBegin,
            SET_A_ITER aEnd,
            SET_B_ITER bBegin
        ){

            while(aBegin != aEnd){

                const auto labelA = *aBegin;
                const auto labelB = *bBegin;

                ++overlaps_[labelA][labelB];
                ++counts_[labelA];
                ++aBegin;
                ++bBegin;
            }
        }


        template<std::size_t DIM, class LABELS_A, class LABELS_B>
        void fill(
            const LABELS_A & arrayA,
            const LABELS_B & arrayB
        ){
            typedef array::StaticArray<int64_t, DIM> Coord;

            Coord shape;
            for(auto d=0; d<DIM; ++d){
                shape[d] = arrayA.shape()[d];
            }
            tools::forEachCoordinate(shape, [&](const Coord coord){
                const LabelType la = xtensor::read(arrayA, coord.asStdArray());
                const LabelType lb = xtensor::read(arrayB, coord.asStdArray());

                ++overlaps_[la][lb];
                ++counts_[la];
            });
        }

        std::vector<CountType> counts_;
        std::vector<MapType>   overlaps_;
    };


} // end namespace nifty::ground_truth
} // end namespace nifty

