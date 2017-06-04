#pragma once

#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_base.hxx"
#include "nifty/graph/optimization/lifted_multicut/lifted_multicut_factory.hxx"
#include "nifty/graph/optimization/lifted_multicut/proposal_generators/proposal_generator_factory_base.hxx"


namespace nifty {
namespace graph {
namespace optimization{
namespace lifted_multicut{







template<class OBJECTIVE>
class PyProposalGeneratorFactoryBase : public ProposalGeneratorFactoryBase<OBJECTIVE> {
public:
    /* Inherit the constructors */
    // using LiftedMulticutFactory<Objective>::LiftedMulticutFactory;
    typedef OBJECTIVE Objective;
    typedef ProposalGeneratorBase<Objective> ProposalGeneratorBaseType;
    /* Trampoline (need one for each virtual function) */
    std::shared_ptr<ProposalGeneratorBaseType> createSharedPtr(const Objective & objective, const size_t numberOfThreads) {
        PYBIND11_OVERLOAD_PURE(
            std::shared_ptr<ProposalGeneratorBaseType>,     /* Return type */
            ProposalGeneratorFactoryBase<Objective>,        /* Parent class */
            createSharedPtr,                                              /* Name of function */
            objective, numberOfThreads                                           /* Argument(s) */
        );
    }
    ProposalGeneratorBaseType * createRawPtr(const Objective & objective, const size_t numberOfThreads) {
        PYBIND11_OVERLOAD_PURE(
            ProposalGeneratorBaseType* ,                    /* Return type */
            ProposalGeneratorFactoryBase<Objective>,        /* Parent class */
            createRawPtr,                                                 /* Name of function */
            objective, numberOfThreads                                           /* Argument(s) */
        );
    }
};


} // namespace lifted_mutlicut
} // namespace optimization
} // namespace graph
} // namespace nifty

