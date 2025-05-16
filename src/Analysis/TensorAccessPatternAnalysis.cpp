#include "Analysis/TensorAccessPatternAnalysis.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"

#define DEBUG_TYPE "tensor-access-pattern-analysis"

STATISTIC(NumSequentialPatterns, "Number of sequential access patterns identified");
STATISTIC(NumStridedPatterns, "Number of strided access patterns identified");
STATISTIC(NumRandomPatterns, "Number of random access patterns identified");
STATISTIC(NumBroadcastPatterns, "Number of broadcast access patterns identified");

using namespace llvm;

namespace llvm {
namespace tensor {

// Unique ID for analysis pass
AnalysisKey TensorAccessPatternAnalysis::Key;

// Implementation of the new pass manager interface
TensorAccessPatternAnalysis::Result TensorAccessPatternAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "Running TensorAccessPatternAnalysis on function " << F.getName() << "\n");

  Result AccessPatterns;

  // Analyze memory access patterns
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        AccessPatterns[Load] = analyzeAccessPattern(Load);
        updateStatistics(AccessPatterns[Load]);
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        AccessPatterns[Store] = analyzeAccessPattern(Store);
        updateStatistics(AccessPatterns[Store]);
      } else if (isTensorOperation(&I)) {
        AccessPatterns[&I] = inferAccessPattern(&I);
        updateStatistics(AccessPatterns[&I]);
      }
    }
  }

  return AccessPatterns;
}

// Helper function to analyze memory access patterns
AccessPattern analyzeAccessPattern(Instruction *I) {
  // This is a placeholder for the actual access pattern analysis logic
  // In a real implementation, this would analyze the instruction's operands and context

  // For now, just return a default pattern
  return AccessPattern::Unknown;
}

// Helper function to infer access patterns for tensor operations
AccessPattern inferAccessPattern(Instruction *I) {
  // This is a placeholder for the actual access pattern inference logic
  // In a real implementation, this would analyze the tensor operation's semantics

  // For now, just return a default pattern
  return AccessPattern::Unknown;
}

// Helper function to update statistics
void updateStatistics(AccessPattern Pattern) {
  switch (Pattern) {
    case AccessPattern::Sequential:
      ++NumSequentialPatterns;
      break;
    case AccessPattern::Strided:
      ++NumStridedPatterns;
      break;
    case AccessPattern::Random:
      ++NumRandomPatterns;
      break;
    case AccessPattern::Broadcast:
      ++NumBroadcastPatterns;
      break;
    default:
      break;
  }
}

// Helper function to identify tensor operations
bool isTensorOperation(Instruction *I) {
  // This is a placeholder for the actual tensor operation identification logic
  // In a real implementation, this would use pattern matching to identify tensor operations

  // For now, just check if the instruction is a call to a function with "tensor" in the name
  if (auto *Call = dyn_cast<CallInst>(I)) {
    if (auto *Callee = Call->getCalledFunction()) {
      if (Callee->getName().contains("tensor")) {
        return true;
      }
    }
  }

  return false;
}

// Legacy pass implementation
char TensorAccessPatternAnalysisWrapperPass::ID = 0;

TensorAccessPatternAnalysisWrapperPass::TensorAccessPatternAnalysisWrapperPass() : FunctionPass(ID) {}

bool TensorAccessPatternAnalysisWrapperPass::runOnFunction(Function &F) {
  // Create a function analysis manager
  FunctionAnalysisManager FAM;

  // Create the analysis pass
  TensorAccessPatternAnalysis TAPA;

  // Run the analysis
  Result = TAPA.run(F, FAM);

  // This is an analysis pass, so it doesn't modify the function
  return false;
}

void TensorAccessPatternAnalysisWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  // This is an analysis pass, so it doesn't modify the function
  AU.setPreservesAll();
}

// Register the legacy pass
static RegisterPass<TensorAccessPatternAnalysisWrapperPass> X("tensor-access-pattern-analysis", "Tensor Access Pattern Analysis Pass");

// Register the pass with the new pass manager
static PassPluginLibraryInfo getTensorAccessPatternAnalysisPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "TensorAccessPatternAnalysis", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerAnalysisRegistrationCallback(
        [](FunctionAnalysisManager &FAM) {
          FAM.registerPass([&] { return TensorAccessPatternAnalysis(); });
        }
      );
    }
  };
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize the pass when added to the pass pipeline on the command line
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getTensorAccessPatternAnalysisPluginInfo();
}

} // namespace tensor
} // namespace llvm
