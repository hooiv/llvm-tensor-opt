#include "Transforms/TensorVectorization.h"
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

#define DEBUG_TYPE "tensor-vectorization"

STATISTIC(NumVectorizedOps, "Number of tensor operations vectorized");

using namespace llvm;

namespace llvm {
namespace tensor {

char TensorVectorizationPass::ID = 0;

// Implementation of the new pass manager interface
PreservedAnalyses TensorVectorizationPass::run(Function &F, FunctionAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "Running TensorVectorization on function " << F.getName() << "\n");

  // Get the results of the tensor access pattern analysis
  auto &TAPA = AM.getResult<TensorAccessPatternAnalysis>(F);

  bool Changed = false;

  // Identify vectorization opportunities
  for (auto &BB : F) {
    for (auto &I : BB) {
      // Skip if this instruction is not a tensor operation
      auto It = TAPA.find(&I);
      if (It == TAPA.end())
        continue;

      // Check if the access pattern is suitable for vectorization
      auto Pattern = It->second;
      if (isSuitableForVectorization(Pattern)) {
        // Perform vectorization
        if (vectorizeTensorOperation(&I)) {
          Changed = true;
          ++NumVectorizedOps;
        }
      }
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// Helper function to check if an access pattern is suitable for vectorization
bool isSuitableForVectorization(AccessPattern Pattern) {
  // Sequential and strided patterns are generally suitable for vectorization
  return Pattern == AccessPattern::Sequential || Pattern == AccessPattern::Strided;
}

// Helper function to vectorize a tensor operation
bool vectorizeTensorOperation(Instruction *I) {
  // This is a placeholder for the actual vectorization logic
  // In a real implementation, this would transform the operation to use vector instructions

  // For now, just return true to indicate that vectorization was performed
  return true;
}

// Legacy pass interface implementation
std::unique_ptr<FunctionPass> createTensorVectorizationPass() {
  return std::make_unique<LegacyTensorVectorizationPass>();
}

// Legacy pass implementation
struct LegacyTensorVectorizationPass : public FunctionPass {
  static char ID;
  LegacyTensorVectorizationPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    auto &TAPA = getAnalysis<TensorAccessPatternAnalysisWrapperPass>().getResult();

    // Implement vectorization logic similar to the new pass manager implementation

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TensorAccessPatternAnalysisWrapperPass>();
    AU.setPreservesCFG();
  }
};

char LegacyTensorVectorizationPass::ID = 0;

// Register the legacy pass
static RegisterPass<LegacyTensorVectorizationPass> X("tensor-vectorization", "Tensor Operation Vectorization Pass");

// Register the pass with the new pass manager
static PassPluginLibraryInfo getTensorVectorizationPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "TensorVectorization", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "tensor-vectorization") {
            FPM.addPass(TensorVectorizationPass());
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
  return getTensorVectorizationPluginInfo();
}

} // namespace tensor
} // namespace llvm
