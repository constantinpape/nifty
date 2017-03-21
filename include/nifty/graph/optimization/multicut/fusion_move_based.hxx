#pragma once
#ifndef NIFTY_GRAPH_OPTIMIZATION_MULTICUT_FUSION_MOVE_BASED_HXX
#define NIFTY_GRAPH_OPTIMIZATION_MULTICUT_FUSION_MOVE_BASED_HXX

#include <mutex>          // std::mutex

#include "nifty/tools/runtime_check.hxx"
#include "nifty/ufd/ufd.hxx"
#include "nifty/graph/detail/adjacency.hxx"
#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/fusion_move.hxx"
#include "nifty/parallel/threadpool.hxx"

namespace nifty{
namespace graph{





    /**
     * @brief      Class for fusion move based inference for the multicut objective
     *             An implementation of \cite beier_15_funsion.
     *             
     *          
     * @tparam     OBJECTIVE  { description }
     */
    template<class PROPPOSAL_GEN>
    class FusionMoveBased : public MulticutBase<typename PROPPOSAL_GEN::Objective >
    {
    public: 

        typedef typename PROPPOSAL_GEN::Objective Objective;
        typedef typename Objective::Graph Graph;
        typedef MulticutBase<Objective> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::VisitorProxy VisitorProxy;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;

        typedef FusionMove<Objective> FusionMoveType;
        typedef PROPPOSAL_GEN ProposalGen;
        typedef typename ProposalGen::Settings ProposalGenSettings;
        typedef typename FusionMoveType::Settings FusionMoveSettings;
        typedef typename Graph:: template EdgeMap<double> EdgeWeights;
    public:

        struct Settings{
            int verbose { 1 };
            int numberOfThreads {-1};
            size_t numberOfIterations {10};
            size_t numberOfParallelProposals {4};
            size_t fuseN{2};
            size_t stopIfNoImprovement{4};
            ProposalGenSettings proposalGenSettings;
            FusionMoveSettings fusionMoveSettings;
        };


        FusionMoveBased(const Objective & objective, const Settings & settings = Settings());
        ~FusionMoveBased();
        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("FusionMoveBased") + PROPPOSAL_GEN::name();
        }

        virtual void weightsChanged(){
            for(size_t i=0; i<pgens_.size(); ++i){
                pgens_[i]->reset();
            }
        }
    private:
        void optimizeParallel(NodeLabels & nodeLabels, VisitorBase * visitor);
        void optimizeSerial(NodeLabels & nodeLabels, VisitorBase * visitor);

        const Objective & objective_;
        const Graph & graph_;
        Settings settings_;
        nifty::parallel::ParallelOptions parallelOptions_;

        std::vector<ProposalGen *>     pgens_;
        std::vector<NodeLabels *>      solBufferIn_;
        std::vector<NodeLabels *>      solBufferOut_;
        std::vector<FusionMoveType * > fusionMoves_;
        NodeLabels * currentBest_;

        nifty::parallel::ThreadPool threadPool_;
    };

    template<class PROPPOSAL_GEN>
    FusionMoveBased<PROPPOSAL_GEN>::
    FusionMoveBased(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        settings_(settings),
        parallelOptions_(settings.numberOfThreads),
        pgens_(),
        threadPool_(parallelOptions_)
    {


        //NIFTY_CHECK(bool(settings_.fusionMoveSettings.mcFactory),"factory is empty");
        const auto nt = parallelOptions_.getActualNumThreads();
        pgens_.resize(nt);
        fusionMoves_.resize(nt);
        solBufferIn_.resize(nt);


        nifty::parallel::parallel_foreach(threadPool_,nt,
            [&](const size_t threadId, const size_t i){
                NIFTY_CHECK_OP(threadId,<,fusionMoves_.size(),"");
                pgens_[i] = new ProposalGen(objective_, settings_.proposalGenSettings, i);
                fusionMoves_[i] = new FusionMoveType(objective_, settings_.fusionMoveSettings);
                solBufferIn_[i] = new NodeLabels(graph_);
        });



    }

    template<class PROPPOSAL_GEN>
    FusionMoveBased<PROPPOSAL_GEN>::
    ~FusionMoveBased(){

        for(size_t i=0; i<parallelOptions_.getActualNumThreads(); ++i){
            delete pgens_[i];
            delete fusionMoves_[i];
            delete solBufferIn_[i];
        }
    }

    template<class PROPPOSAL_GEN>
    void FusionMoveBased<PROPPOSAL_GEN>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        if(graph_.numberOfEdges()==0){
            if(visitor!=nullptr){
                visitor->begin(this);
                visitor->end(this);
            }
        }
        else{
            if(parallelOptions_.getNumThreads() > 0){ // TODO FIXME isn't > 1 more appropriate here ?!
                this->optimizeParallel(nodeLabels, visitor);
            }
            else{
                this->optimizeSerial(nodeLabels, visitor);
            }
        }
    }


    template<class PROPPOSAL_GEN>
    void FusionMoveBased<PROPPOSAL_GEN>::
    optimizeParallel(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){

        currentBest_ = &nodeLabels;
        if(visitor!=nullptr){
            visitor->addLogNames({"IterationWithoutImprovement"});
            visitor->begin(this);
        }
        auto & currentBest = nodeLabels;
        auto bestEnergy = objective_.evalNodeLabels(currentBest);
    

        std::mutex mtx;


        std::vector<NodeLabels> proposals;
        auto iterWithoutImprovement = 0;
        for(auto iter=0; iter<settings_.numberOfIterations; ++iter){
            //std::cout<<"iter "<<iter<<"\n";
            const auto oldBestEnergy = bestEnergy;
            proposals.clear();
            // generate the proposals and fuse with the current best
            //std::cout<<"generate "<<settings_.numberOfParallelProposals<<" proposals\n";
            nifty::parallel::parallel_foreach(threadPool_,
                settings_.numberOfParallelProposals,
                [&](const size_t threadId, int proposalIndex){
                    NIFTY_CHECK_OP(threadId,<,fusionMoves_.size(),"");
                    // 
                    auto & pgen = *pgens_[threadId];
                    auto & proposal = *solBufferIn_[threadId];
                    //std::cout<<"generate\n";
                    pgen.generate(currentBest, proposal);
                    //std::cout<<"generate done\n";
                    // evaluate the energy of the proposal
                    const auto eProposal = objective_.evalNodeLabels(proposal);
                    
                    
                
                    if(bestEnergy < -0.00001 || iter != 0){  // fuse with current best
                        //std::cout<<"a\n";

                        mtx.lock();
                        NodeLabels bestCopy = currentBest;
                        auto eBestCopy = objective_.evalNodeLabels(bestCopy);
                        mtx.unlock(); 

                        NodeLabels res(graph_);
                        auto & fm = *(fusionMoves_[threadId]);
                        fm.fuse( {&proposal, &currentBest}, &res);
                        auto eFuse = objective_.evalNodeLabels(res);

                        mtx.lock();
                        if(eFuse < eBestCopy){
                            currentBest = res;
                            bestEnergy = eFuse;
                            //proposals.push_back(res);
                        }
                        
                        mtx.unlock();
                    }
                    else{  // just keep this one and do not fuse with current best
                        //std::cout<<"a\n";
                        mtx.lock();
                        NodeLabels bestCopy = currentBest;
                        auto eBestCopy = objective_.evalNodeLabels(bestCopy);
                        if(eProposal < eBestCopy){
                            bestEnergy = eProposal;
                            currentBest = proposal;
                        }
                        proposals.push_back(proposal);
                        mtx.unlock();       
                    }   
                }
            );
          
            std::cout<<"Generated "<<proposals.size() << " in parallel." << "\n";
            // recursive thing
            
            /*
            std::vector<NodeLabels> proposals2;
            size_t nFuse = settings_.fuseN;
            
            if(!proposals.empty()){
                while(proposals.size()!= 1){
                    //std::cout<<" aaa \n";
                    NIFTY_CHECK_OP(proposals.size(),>=,2,"");
                    nFuse = std::min(nFuse, proposals.size());

                    auto pSize = proposals.size() / nFuse;
                    if( proposals.size() % nFuse != 0){
                        ++pSize;
                    }

                    nifty::parallel::parallel_foreach(threadPool_, 
                                                      pSize,
                    [&](const int threadId, const int ii){

                
                        auto i = ii*nFuse;

                        std::vector<const NodeLabels*> toFuse;
                        for(size_t j=0; j<nFuse; ++j){
                            auto k = i + j < proposals.size() ? i+j : i+j - proposals.size();
                            toFuse.push_back(&proposals[k]);
                        }

                        // here we start to fuse them
                        NodeLabels res(graph_);
                        auto & fm = *(fusionMoves_[threadId]);

                        fm.fuse(toFuse, &res);
                
                        mtx.lock();
                        proposals2.push_back(res);
                        mtx.unlock();
                    });

                    ////std::cout<<"proposals 2 size "<<proposals2.size()<<"\n\n";
                    proposals = proposals2;
                    proposals2.clear();
                }
                currentBest = proposals[0];
            }
            //std::cout<<"doish\n";
            bestEnergy = objective_.evalNodeLabels(currentBest);
            */

            // call the visitor and see if we need to continue
            if(visitor!= nullptr){
                visitor->setLogValue(0,iterWithoutImprovement);
                if(!visitor->visit(this))
                    break;
            }
            if(bestEnergy < oldBestEnergy){
                iterWithoutImprovement = 0;
            }
            else{
                ++iterWithoutImprovement;
            }
            if(iterWithoutImprovement > settings_.stopIfNoImprovement){
                break;
            }
        }

        //for(auto node : graph_.nodes())
        //    nodeLabels[node] = currentBest[node];
        if(visitor!=nullptr)
            visitor->end(this);
    }

    template<class PROPPOSAL_GEN>
    void FusionMoveBased<PROPPOSAL_GEN>::
    optimizeSerial(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){
        VisitorProxy visitorProxy(visitor);

        currentBest_ = &nodeLabels;
       
        visitorProxy.addLogNames({"IterationWithoutImprovement"});
        visitorProxy.begin(this);
        
        auto & currentBest = nodeLabels;
        auto bestEnergy = objective_.evalNodeLabels(currentBest);
    




        std::vector<NodeLabels> proposals;
        size_t iterWithoutImprovement = 0;
        for(auto iter=0; iter<settings_.numberOfIterations; ++iter){
            const auto oldBestEnergy = bestEnergy;
        
            auto & pgen = *pgens_[0];
            auto & proposal = *solBufferIn_[0];
            pgen.generate(currentBest, proposal);

            if(bestEnergy>=-0.000000001 && iter==0){
                currentBest = proposal;
                bestEnergy = objective_.evalNodeLabels(currentBest);
                iterWithoutImprovement = 0;
            }
            else{
                NodeLabels res(graph_);
                auto & fm = *(fusionMoves_[0]);
                fm.fuse( {&proposal, &currentBest}, &res);
                auto eFuse = objective_.evalNodeLabels(res);
                if(eFuse<bestEnergy){
                    bestEnergy = eFuse;
                    currentBest = res;
                    iterWithoutImprovement = 0;
                }   
                else{
                    ++iterWithoutImprovement;
                }
            }

            if(iterWithoutImprovement > settings_.stopIfNoImprovement){
                break;
            }

            visitorProxy.setLogValue(0,iterWithoutImprovement);
            if(!visitorProxy.visit(this))
                break;
        }

        //for(auto node : graph_.nodes())
        //    nodeLabels[node] = currentBest[node];
        
        visitorProxy.end(this);
    }

    template< class PROPPOSAL_GEN>
    const typename FusionMoveBased<PROPPOSAL_GEN>::Objective &
    FusionMoveBased<PROPPOSAL_GEN>::
    objective()const{
        return objective_;
    }



} // namespace nifty::graph
} // namespace nifty

#endif  // NIFTY_GRAPH_OPTIMIZATION_MULTICUT_FUSION_MOVE_BASED_HXX
