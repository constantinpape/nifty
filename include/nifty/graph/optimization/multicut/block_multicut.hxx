#pragma once


#include "nifty/tools/runtime_check.hxx"

#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_factory.hxx"
#include "nifty/graph/optimization/multicut/multicut_objective.hxx"






namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

   


    template<class OBJECTIVE>
    class BlockMulticut : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef OBJECTIVE ObjectiveType;
        typedef typename ObjectiveType::WeightType WeightType;
        typedef MulticutBase<ObjectiveType> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::VisitorProxy VisitorProxy;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;
        typedef typename ObjectiveType::Graph Graph;
        typedef typename ObjectiveType::GraphType GraphType;
        typedef typename ObjectiveType::WeightsMap WeightsMap;
        typedef typename GraphType:: template EdgeMap<uint8_t> IsDirtyEdge;


        typedef MulticutFactoryBase<ObjectiveType>  McFactoryBase;


    
    public:

        struct Settings{
            std::shared_ptr<McFactoryBase> multicutFactory;
        };

        virtual ~BlockMulticut(){
            
        }
        BlockMulticut(const Objective & objective, const Settings & settings = Settings());


        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("BlockMulticut");
        }
        virtual void weightsChanged(){ 
        }
        virtual double currentBestEnergy() {
           return currentBestEnergy_;
        }
    private:


        const Objective & objective_;
        Settings settings_;
        NodeLabels * currentBest_;
        double currentBestEnergy_;
    
    };

    
    template<class OBJECTIVE>
    BlockMulticut<OBJECTIVE>::
    BlockMulticut(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        settings_(settings),
        currentBest_(nullptr),
        currentBestEnergy_(std::numeric_limits<double>::infinity())
    {

    }

    template<class OBJECTIVE>
    void BlockMulticut<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){  


        
        VisitorProxy visitorProxy(visitor);
        currentBest_ = &nodeLabels;
        currentBestEnergy_ = objective_.evalNodeLabels(nodeLabels);
        
        visitorProxy.begin(this);

  
        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    const typename BlockMulticut<OBJECTIVE>::Objective &
    BlockMulticut<OBJECTIVE>::
    objective()const{
        return objective_;
    }


} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

