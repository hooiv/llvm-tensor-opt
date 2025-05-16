#include "Transforms/TensorParallelization.h"
#include "Analysis/TensorAccessPatternAnalysis.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"

#define DEBUG_TYPE "tensor-parallelization"

STATISTIC(NumParallelizedOps, "Number of tensor operations parallelized");

using namespace llvm;

namespace llvm {
namespace tensor {

char TensorParallelizationPass::ID = 0;

// Implementation of the new pass manager interface
PreservedAnalyses TensorParallelizationPass::run(Function &F, FunctionAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "Running TensorParallelization on function " << F.getName() << "\n");

  // Get the results of the tensor access pattern analysis
  auto &TAPA = AM.getResult<TensorAccessPatternAnalysis>(F);

  bool Changed = false;

  // Identify parallelization opportunities
  for (auto &BB : F) {
    for (auto &I : BB) {
      // Skip if this instruction is not a tensor operation
      auto It = TAPA.find(&I);
      if (It == TAPA.end())
        continue;

      // Check if the access pattern is suitable for parallelization
      auto Pattern = It->second;
      if (isSuitableForParallelization(Pattern)) {
        // Perform parallelization
        if (parallelizeTensorOperation(&I)) {
          Changed = true;
          ++NumParallelizedOps;
        }
      }
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// Helper function to check if an access pattern is suitable for parallelization
bool isSuitableForParallelization(AccessPattern Pattern) {
  // Most patterns can be parallelized on a GPU
  return Pattern != AccessPattern::Unknown;
}

// Helper function to parallelize a tensor operation
bool parallelizeTensorOperation(Instruction *I) {
  // This is a placeholder for the actual parallelization logic
  // In a real implementation, this would transform the operation to use GPU parallelism

  // For now, just return true to indicate that parallelization was performed
  return true;
}

// Legacy pass interface implementation
std::unique_ptr<FunctionPass> createTensorParallelizationPass() {
  return std::make_unique<LegacyTensorParallelizationPass>();
}

// Legacy pass implementation
struct LegacyTensorParallelizationPass : public FunctionPass {
  static char ID;
  LegacyTensorParallelizationPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    auto &TAPA = getAnalysis<TensorAccessPatternAnalysisWrapperPass>().getResult();

    // Implement parallelization logic similar to the new pass manager implementation

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TensorAccessPatternAnalysisWrapperPass>();
    AU.setPreservesCFG();
  }
};

char LegacyTensorParallelizationPass::ID = 0;

// Register the legacy pass
static RegisterPass<LegacyTensorParallelizationPass> X("tensor-parallelization", "Tensor Operation Parallelization Pass");

// Register the pass with the new pass manager
static PassPluginLibraryInfo getTensorParallelizationPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "TensorParallelization", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "tensor-parallelization") {
            FPM.addPass(TensorParallelizationPass());
            return true;
          }
          return false;
        }
      );
    }
  };
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize the pass when added to the pass pipeline on the command line
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getTensorParallelizationPluginInfo();
}

} // namespace tensor
} // namespace llvm
