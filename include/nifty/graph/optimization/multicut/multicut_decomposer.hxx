#pragma once

#include "nifty/tools/runtime_check.hxx"
#include "nifty/graph/components.hxx"

#include "nifty/graph/optimization/multicut/multicut_base.hxx"
#include "nifty/graph/optimization/multicut/multicut_factory.hxx"
#include "nifty/graph/optimization/multicut/multicut_objective.hxx"
#include "nifty/graph/undirected_list_graph.hxx"

namespace nifty{
namespace graph{
namespace optimization{
namespace multicut{

    template<class OBJECTIVE>
    class MulticutDecomposer : public MulticutBase<OBJECTIVE>
    {
    public: 

        typedef OBJECTIVE Objective;
        typedef typename Objective::WeightType WeightType;
        typedef MulticutBase<OBJECTIVE> Base;
        typedef typename Base::VisitorBase VisitorBase;
        typedef typename Base::VisitorProxy VisitorProxy;
        typedef typename Base::EdgeLabels EdgeLabels;
        typedef typename Base::NodeLabels NodeLabels;
        typedef typename Objective::Graph Graph;
        typedef typename Objective::WeightsMap WeightsMap;


        typedef MulticutFactoryBase<Objective>         FactoryBase;
       
    private:
        typedef ComponentsUfd<Graph> Components;
        
        
    public:
        typedef UndirectedGraph<>                            SubmodelGraph;
        typedef MulticutObjective<SubmodelGraph, WeightType> SubmodelObjective;
        typedef MulticutBase<SubmodelObjective>              SubmodelMulticutBase;
        typedef MulticutFactoryBase<SubmodelObjective>       SubmodelFactoryBase;
        typedef typename SubmodelMulticutBase::NodeLabels    SubmodelNodeLabels;

    public:

        struct Settings{
            std::shared_ptr<SubmodelFactoryBase> submodelFactory;
            std::shared_ptr<FactoryBase>         fallthroughFactory;
            
        };

        virtual ~MulticutDecomposer(){
            
        }
        MulticutDecomposer(const Objective & objective, const Settings & settings = Settings());


        virtual void optimize(NodeLabels & nodeLabels, VisitorBase * visitor);
        virtual const Objective & objective() const;


        virtual const NodeLabels & currentBestNodeLabels( ){
            return *currentBest_;
        }

        virtual std::string name()const{
            return std::string("MulticutDecomposer");
        }
        virtual void weightsChanged(){ 
        }
        
    private:

        struct SubgraphWithCut {
            SubgraphWithCut(const WeightsMap & weights)
                :   weights_(weights)
            {}
            bool useNode(const size_t v) const
                { return true; }
            bool useEdge(const size_t e) const
            {
                return weights_[e] > 0.0;
            }

            const WeightsMap & weights_;
        };





        const Objective & objective_;
        const Graph & graph_;
        const WeightsMap & weights_;

        Components components_;
        NodeLabels * currentBest_;

        Settings settings_;
    };

    
    template<class OBJECTIVE>
    MulticutDecomposer<OBJECTIVE>::
    MulticutDecomposer(
        const Objective & objective, 
        const Settings & settings
    )
    :   objective_(objective),
        graph_(objective.graph()),
        weights_(objective_.weights()),
        components_(graph_),
        settings_(settings)
    {
        if(!bool(settings_.fallthroughFactory)){
            throw std::runtime_error("MulticutDecomposer Settings: fallthroughFactory may not be empty!");
        }
        if(!bool(settings_.submodelFactory)){
            throw std::runtime_error("MulticutDecomposer Settings: submodelFactory may not be empty!");
        }
    }

    template<class OBJECTIVE>
    void MulticutDecomposer<OBJECTIVE>::
    optimize(
        NodeLabels & nodeLabels,  VisitorBase * visitor
    ){  

        
        VisitorProxy visitorProxy(visitor);
        //visitorProxy.addLogNames({"violatedConstraints"});
        currentBest_ = &nodeLabels;
        
        visitorProxy.begin(this);


        // build the connected components
        NodeLabels denseLabels(graph_);
        const auto nComponents = components_.build(SubgraphWithCut(weights_));
        std::vector<size_t> componentsSize(nComponents,0);
        components_.denseRelabeling(denseLabels, componentsSize);


        visitorProxy.printLog(nifty::logging::LogLevel::INFO, 
            std::string("model decomposes in ")+std::to_string(nComponents));


        // build the sub objectives in the case 
        // the thing decomposes
        if(nComponents >= 2){
            

            visitorProxy.clearLogNames();
            visitorProxy.addLogNames({std::string("modelSize")});



            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "alloc subgraphs");
            // first pass :
            // - allocate the sub graphs
            // - sparse to dense
            std::vector<SubmodelGraph *> subGraphVec(nComponents);
            for(size_t i=0; i<nComponents; ++i){
                NIFTY_CHECK_OP(componentsSize[i],>,0,"");
                if(componentsSize[i]>1)
                    subGraphVec[i] = new SubmodelGraph(componentsSize[i]);
            }


            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "global to local mappin");
            // map from global variables to
            // subproblem variables
            std::vector< std::unordered_map<uint64_t, uint64_t>  > nodeToSubNodeVec(nComponents);
            for(const auto node : graph_.nodes()){
                const auto subgraphId = denseLabels[node];
                auto & nodeToSubNode = nodeToSubNodeVec[subgraphId];
                const auto subVarId = nodeToSubNode.size();
                nodeToSubNode[node] = subVarId;
            }


            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "add edges");
            // add edges to subproblems
            for(const auto edge : graph_.edges()){

                const auto u = graph_.u(edge);
                const auto v = graph_.v(edge);
                const auto lu = denseLabels[u];

                if(componentsSize[lu]>1){
                    if(lu == denseLabels[v]){

                        NIFTY_CHECK_OP(lu,<,nComponents,"");

                        const auto & nodeToSubNode = nodeToSubNodeVec[lu];

                        const auto findU = nodeToSubNode.find(u);
                        const auto findV = nodeToSubNode.find(v);

                        NIFTY_CHECK(findU != nodeToSubNode.end(),"");
                        NIFTY_CHECK(findV != nodeToSubNode.end(),"");

                        const auto su = findU->second;
                        const auto sv = findV->second;

                        NIFTY_CHECK_OP(su,<,subGraphVec[lu]->numberOfNodes(),"");
                        NIFTY_CHECK_OP(sv,<,subGraphVec[lu]->numberOfNodes(),"");

                        subGraphVec[lu]->insertEdge(su,sv);
                    }
                }
            }

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "build sub-objectives");
            // build the sub mc objectives
            std::vector<SubmodelObjective *> subObjectiveVec(nComponents);
            for(size_t i=0; i<nComponents; ++i){
                if(componentsSize[i]>1)
                    subObjectiveVec[i] = new SubmodelObjective(*subGraphVec[i]);
            }

            for(const auto edge : graph_.edges()){
                const auto u = graph_.u(edge);
                const auto v = graph_.v(edge);
                const auto lu = denseLabels[u];
                if(componentsSize[lu]>1){
                    if(lu == denseLabels[v]){
                        const auto & nodeToSubNode = nodeToSubNodeVec[lu];
                        const auto su = nodeToSubNode.find(u)->second;
                        const auto sv = nodeToSubNode.find(v)->second;
                        const auto subEdge = subGraphVec[lu]->findEdge(su,sv);
                        subObjectiveVec[lu]->weights()[subEdge] = weights_[edge];
                    }
                }
            }

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "optimize subproblems");
            // optimize the subproblems
            // and delete stuff we do not need anymore
            std::vector<SubmodelNodeLabels *> subNodeLabelsVec(nComponents);

            // //////////////////////////////////////////////
            // solving and partial cleanup
            // //////////////////////////////////////////////
            for(size_t i=0; i<nComponents; ++i){
                if(componentsSize[i]>1){
                    const auto & subObj = *subObjectiveVec[i];
                    const auto & subGraph = *subGraphVec[i];
                    const auto nSubGraphNodes = subGraph.numberOfNodes();
                    std::cout<<"#Nodes "<<nSubGraphNodes<<" "<<float(nSubGraphNodes)/float(graph_.numberOfNodes())<<"\n";
                  
                    subNodeLabelsVec[i] = new SubmodelNodeLabels(subGraph);

                    // create solver and optimize 
                    auto subSolver = settings_.submodelFactory->createRawPtr(subObj);
                    subSolver->optimize(*subNodeLabelsVec[i], nullptr);
                    delete subSolver;
                    delete subObjectiveVec[i];
                    delete subGraphVec[i];
                }
            }

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "map sub to global");
            // map from the sub solutions to global solution
            {
                nifty::ufd::Ufd< > ufd(graph_.nodeIdUpperBound()+1);
                for(const auto edge : graph_.edges()){

                    const auto u = graph_.u(edge);
                    const auto v = graph_.v(edge);
                    const auto lu = denseLabels[u];

                    if(componentsSize[lu]>1){
                        if(lu == denseLabels[v]){

                            const auto & nodeToSubNode = nodeToSubNodeVec[lu];
                            const auto & subNodeLabels = *subNodeLabelsVec[lu];
                            const auto su = nodeToSubNode.find(u)->second;
                            const auto sv = nodeToSubNode.find(v)->second;
                            if(subNodeLabels[su] == subNodeLabels[sv]){
                                ufd.merge(u, v);
                            }
                        }
                    }
                }
                
                for(const auto node : graph_.nodes()){
                    nodeLabels[node] = ufd.find(node);
                }
            }

            //visitorProxy.printLog(nifty::logging::LogLevel::INFO, "cleanup");
            // final cleanup
            for(size_t i=0; i<nComponents; ++i){
                if(componentsSize[i]>1){
                    delete subNodeLabelsVec[i];
                }
            }
            visitorProxy.clearLogNames();
        }
        else{
            auto solverPtr = settings_.fallthroughFactory->createRawPtr(objective_);
            // TODO handle visitor:
            // Problem: if we just pass the
            // visitor begin and end are called twice
            solverPtr->optimize(nodeLabels, nullptr);
            delete solverPtr;
        }

        visitorProxy.end(this);
    }

    template<class OBJECTIVE>
    const typename MulticutDecomposer<OBJECTIVE>::Objective &
    MulticutDecomposer<OBJECTIVE>::
    objective()const{
        return objective_;
    }


} // namespace nifty::graph::optimization::multicut
} // namespace nifty::graph::optimization
} // namespace nifty::graph
} // namespace nifty

