#include "AutoTuning/AutoTuner.h"
#include "Transforms/TensorFusion.h"
#include "Transforms/TensorVectorization.h"
#include "Transforms/TensorParallelization.h"
#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorAccessPatternAnalysis.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "auto-tuner"

using namespace llvm;
using namespace llvm::tensor;

AutoTuner::AutoTuner(bool UseML)
  : CostModelImpl(createCostModel(UseML)) {
}

bool AutoTuner::tune(Function &F, FunctionAnalysisManager &AM) {
  // Get the best optimization strategy for this function
  OptimizationStrategy BestStrategy = getBestStrategy(F);
  
  LLVM_DEBUG(dbgs() << "AutoTuner: Best strategy for function " << F.getName()
                    << " is " << static_cast<int>(BestStrategy) << "\n");
  
  // Apply the best optimization strategy
  return applyStrategy(F, AM, BestStrategy);
}

OptimizationStrategy AutoTuner::getBestStrategy(const Function &F) const {
  return CostModelImpl->getBestStrategy(F);
}

bool AutoTuner::applyStrategy(Function &F, FunctionAnalysisManager &AM, OptimizationStrategy Strategy) {
  bool Changed = false;
  
  // Apply the optimization passes based on the strategy
  switch (Strategy) {
    case OptimizationStrategy::None:
      // No optimizations
      break;
      
    case OptimizationStrategy::Fusion:
      // Apply fusion only
      {
        TensorFusionPass FusionPass;
        PreservedAnalyses PA = FusionPass.run(F, AM);
        Changed = !PA.areAllPreserved();
      }
      break;
      
    case OptimizationStrategy::Vectorization:
      // Apply vectorization only
      {
        TensorVectorizationPass VectorizationPass;
        PreservedAnalyses PA = VectorizationPass.run(F, AM);
        Changed = !PA.areAllPreserved();
      }
      break;
      
    case OptimizationStrategy::Parallelization:
      // Apply parallelization only
      {
        TensorParallelizationPass ParallelizationPass;
        PreservedAnalyses PA = ParallelizationPass.run(F, AM);
        Changed = !PA.areAllPreserved();
      }
      break;
      
    case OptimizationStrategy::FusionAndVectorization:
      // Apply fusion and vectorization
      {
        TensorFusionPass FusionPass;
        PreservedAnalyses PA1 = FusionPass.run(F, AM);
        
        TensorVectorizationPass VectorizationPass;
        PreservedAnalyses PA2 = VectorizationPass.run(F, AM);
        
        Changed = !PA1.areAllPreserved() || !PA2.areAllPreserved();
      }
      break;
      
    case OptimizationStrategy::FusionAndParallelization:
      // Apply fusion and parallelization
      {
        TensorFusionPass FusionPass;
        PreservedAnalyses PA1 = FusionPass.run(F, AM);
        
        TensorParallelizationPass ParallelizationPass;
        PreservedAnalyses PA2 = ParallelizationPass.run(F, AM);
        
        Changed = !PA1.areAllPreserved() || !PA2.areAllPreserved();
      }
      break;
      
    case OptimizationStrategy::VectorizationAndParallelization:
      // Apply vectorization and parallelization
      {
        TensorVectorizationPass VectorizationPass;
        PreservedAnalyses PA1 = VectorizationPass.run(F, AM);
        
        TensorParallelizationPass ParallelizationPass;
        PreservedAnalyses PA2 = ParallelizationPass.run(F, AM);
        
        Changed = !PA1.areAllPreserved() || !PA2.areAllPreserved();
      }
      break;
      
    case OptimizationStrategy::All:
      // Apply all optimizations
      {
        TensorFusionPass FusionPass;
        PreservedAnalyses PA1 = FusionPass.run(F, AM);
        
        TensorVectorizationPass VectorizationPass;
        PreservedAnalyses PA2 = VectorizationPass.run(F, AM);
        
        TensorParallelizationPass ParallelizationPass;
        PreservedAnalyses PA3 = ParallelizationPass.run(F, AM);
        
        Changed = !PA1.areAllPreserved() || !PA2.areAllPreserved() || !PA3.areAllPreserved();
      }
      break;
  }
  
  return Changed;
}
