#include "Transforms/TensorFusion.h"
#include "Analysis/TensorDataFlowAnalysis.h"
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

#define DEBUG_TYPE "tensor-fusion"

STATISTIC(NumFusedOps, "Number of tensor operations fused");

using namespace llvm;

namespace llvm {
namespace tensor {

char TensorFusionPass::ID = 0;

// Implementation of the new pass manager interface
PreservedAnalyses TensorFusionPass::run(Function &F, FunctionAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "Running TensorFusion on function " << F.getName() << "\n");

  // Get the results of the tensor data flow analysis
  auto &TDFA = AM.getResult<TensorDataFlowAnalysis>(F);
  auto &TAPA = AM.getResult<TensorAccessPatternAnalysis>(F);

  bool Changed = false;

  // Identify fusion opportunities
  for (auto &BB : F) {
    for (auto I = BB.begin(), E = BB.end(); I != E; ++I) {
      Instruction *Inst = &*I;

      // Skip if this instruction is not a tensor operation
      if (TDFA.find(Inst) == TDFA.end())
        continue;

      // Get the data flow for this instruction
      auto &DataFlow = TDFA.find(Inst)->second;

      // Skip if there are no dependent operations
      if (DataFlow.empty())
        continue;

      // Check if we can fuse this operation with its dependent operations
      for (auto *Dep : DataFlow) {
        // Skip if the dependent operation is not in the same basic block
        if (Dep->getParent() != &BB)
          continue;

        // Check if the access patterns are compatible for fusion
        auto InstPattern = TAPA.find(Inst)->second;
        auto DepPattern = TAPA.find(Dep)->second;

        if (isCompatibleForFusion(InstPattern, DepPattern)) {
          // Perform fusion
          if (fuseTensorOperations(Inst, Dep)) {
            Changed = true;
            ++NumFusedOps;

            // Update the iterator to avoid processing the fused operation
            I = BB.begin();
            break;
          }
        }
      }
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// Helper function to check if two access patterns are compatible for fusion
bool isCompatibleForFusion(AccessPattern A, AccessPattern B) {
  // Sequential and strided patterns are generally compatible
  if ((A == AccessPattern::Sequential || A == AccessPattern::Strided) &&
      (B == AccessPattern::Sequential || B == AccessPattern::Strided))
    return true;

  // Broadcast pattern can be fused with any other pattern
  if (A == AccessPattern::Broadcast || B == AccessPattern::Broadcast)
    return true;

  // Random access patterns are generally not compatible for fusion
  if (A == AccessPattern::Random || B == AccessPattern::Random)
    return false;

  // Unknown patterns are conservatively not fused
  return false;
}

// Helper function to fuse two tensor operations
bool fuseTensorOperations(Instruction *First, Instruction *Second) {
  // This is a placeholder for the actual fusion logic
  // In a real implementation, this would create a new fused operation
  // and replace the original operations

  // For now, just return true to indicate that fusion was performed
  return true;
}

// Legacy pass interface implementation
std::unique_ptr<FunctionPass> createTensorFusionPass() {
  return std::make_unique<LegacyTensorFusionPass>();
}

// Legacy pass implementation
struct LegacyTensorFusionPass : public FunctionPass {
  static char ID;
  LegacyTensorFusionPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    auto &TDFA = getAnalysis<TensorDataFlowAnalysisWrapperPass>().getResult();
    auto &TAPA = getAnalysis<TensorAccessPatternAnalysisWrapperPass>().getResult();

    // Implement fusion logic similar to the new pass manager implementation

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TensorDataFlowAnalysisWrapperPass>();
    AU.addRequired<TensorAccessPatternAnalysisWrapperPass>();
    AU.setPreservesCFG();
  }
};

char LegacyTensorFusionPass::ID = 0;

// Register the legacy pass
static RegisterPass<LegacyTensorFusionPass> X("tensor-fusion", "Tensor Operation Fusion Pass");

// Register the pass with the new pass manager
static PassPluginLibraryInfo getTensorFusionPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "TensorFusion", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "tensor-fusion") {
            FPM.addPass(TensorFusionPass());
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
  return getTensorFusionPluginInfo();
}

} // namespace tensor
} // namespace llvm
