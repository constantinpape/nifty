#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"
#include "nifty/graph/paths.hxx"
#include "nifty/graph/optimization/mincut/mincut_base.hxx"
#include "nifty/graph/detail/contiguous_indices.hxx"


#include "QPBO.h"


namespace nifty{
namespace graph{
namespace optimization{
namespace mincut{

    template<class OBJECTIVE>
    class MincutQpbo : public MincutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef MincutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::VisitorProxy VisitorProxy;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;
        typedef typename Objective::Graph Graph;

    private:
        typedef float QpboValueType;
        typedef detail_graph::NodeIndicesToContiguousNodeIndices<Graph> DenseIds;



    public:

        struct Settings{
            bool improve{true};
            //bool guaranteeNoParallelEdges {false};
        };

        virtual ~MincutQpbo(){

        }
        MincutQpbo(const Objective & objective, const Settings & settings = Settings());


        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }
        virtual double currentBestEnergy() {
           return currentBestEnergy_;
        }
        virtual std::string name()const{
            return std::string("MincutQpbo");
        }

        virtual void weightsChanged(){  
            this->qpbo_.Reset();
            this->initializeQpbo();
        }
        
    private:

        void initializeQpbo();


        void repairSolution(NodeLabels & nodeLabels);


        size_t addCycleInequalities();
        void addThreeCyclesConstraintsExplicitly();

        const Objective & objective_;
        const Graph & graph_;

        // zero overhead lookup for graphs which have already
        // dense ids (only merge graph does not have dense ids)
        DenseIds denseNodeIds_;
        Settings settings_;
        NodeLabels * currentBest_;
        double currentBestEnergy_;
        QPBO<QpboValueType> qpbo_;
    };

    
    template<class OBJECTIVE>
    MincutQpbo<OBJECTIVE>::
    MincutQpbo(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        denseNodeIds_(objective.graph()),
        settings_(settings),
        currentBest_(nullptr),
        currentBestEnergy_(std::numeric_limits<double>::infinity()),
        qpbo_(objective.graph().numberOfNodes(), objective.graph().numberOfEdges())

    {
        // initialize qpbo
        this->initializeQpbo();
 
    }

    template<class OBJECTIVE>
    void MincutQpbo<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){  

        VisitorProxy visitorProxy(visitor);
        visitorProxy.begin(this);

        currentBest_ = &nodeLabels;

        // little lambda to read sol from qpbo
        // and compute energys
        auto readQpboSol = [&](){
            uint64_t nodeD = 0;
            for(const auto node : graph_.nodes()){
                const auto triLabel = qpbo_.GetLabel(nodeD);
                nodeLabels[node] =  triLabel == 0 ? 0 : (triLabel == 1 ? 1 : 0);
                ++nodeD;
            }
            auto e = qpbo_.ComputeTwiceEnergy()/2.0;
            auto firstNodeLabel = 0 ;
            nodeD = 0;
            for(const auto node : graph_.nodes()){
                const auto triLabel = qpbo_.GetLabel(nodeD);
                const auto label = (triLabel == 0 ? 0 : (triLabel == 1 ? 1 : 0));
                break;
            }
            if(graph_.numberOfNodes()>0){
                if (firstNodeLabel == 1){
                    e -= 1000000.0;
                }
            }
            return e;
        };

        // solve
        //visitorProxy.printLog(nifty::logging::LogLevel::DEBUG, "Solve Qpbo");
        qpbo_.Solve();
        //qpbo_.Improve();
        

        // improve
        if(settings_.improve){
            if(bool(visitorProxy)){
                currentBestEnergy_ = readQpboSol();
                visitorProxy.visit(this);
            }
            srand(42);
            //visitorProxy.printLog(nifty::logging::LogLevel::DEBUG, "Improve Qpbo");
            qpbo_.Improve();
        }

        currentBestEnergy_ = readQpboSol();
       
        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    inline const typename MincutQpbo<OBJECTIVE>::Objective &
    MincutQpbo<OBJECTIVE>::
    objective()const{
        return objective_;
    }


    template<class OBJECTIVE>
    inline void MincutQpbo<OBJECTIVE>::
    initializeQpbo(){

        
        qpbo_.AddNode(graph_.numberOfNodes());
        qpbo_.SetMaxEdgeNum(graph_.numberOfEdges());
        
        if(graph_.numberOfNodes()>0)
            qpbo_.AddUnaryTerm(0, 0.0, 1000000.0);
        for(const auto edge : graph_.edges()){
            const auto uD = denseNodeIds_[graph_.u(edge)];
            const auto vD = denseNodeIds_[graph_.v(edge)];
            NIFTY_CHECK_OP(uD,<,graph_.numberOfNodes(),"");
            NIFTY_CHECK_OP(vD,<,graph_.numberOfNodes(),"");
            const auto w  = objective_.weights()[edge];
            qpbo_.AddPairwiseTerm(uD,vD,0.0,w,w,0.0);
        }
    }

} // namespace nifty::graph::optimization::mincut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty
