#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorOperationRegistry.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"

#define DEBUG_TYPE "tensor-data-flow-analysis"

STATISTIC(NumTensorOps, "Number of tensor operations identified");
STATISTIC(NumDataFlowEdges, "Number of data flow edges identified");

using namespace llvm;

namespace llvm {
namespace tensor {

// Forward declarations for helper functions
bool isTensorOperation(Instruction *I);
bool hasTensorAccessPattern(Instruction *I);
bool hasMultidimensionalAccess(Value *V);

// Unique ID for analysis pass
AnalysisKey TensorDataFlowAnalysis::Key;

// Implementation of the new pass manager interface
TensorDataFlowAnalysis::Result TensorDataFlowAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "Running TensorDataFlowAnalysis on function " << F.getName() << "\n");

  Result DataFlow;

  // Identify tensor operations
  std::vector<Instruction*> TensorOps;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (isTensorOperation(&I)) {
        TensorOps.push_back(&I);
        DataFlow[&I] = std::vector<Instruction*>();
        ++NumTensorOps;
      }
    }
  }

  // Build data flow graph
  for (auto *Op : TensorOps) {
    for (auto *User : Op->users()) {
      if (auto *UserInst = dyn_cast<Instruction>(User)) {
        if (isTensorOperation(UserInst)) {
          DataFlow[Op].push_back(UserInst);
          ++NumDataFlowEdges;
        }
      }
    }
  }

  return DataFlow;
}

// Helper function to identify tensor operations
inline bool isTensorOperation(Instruction *I) {
  // Use the tensor operation registry to identify tensor operations
  auto Op = TensorOperationRegistry::getInstance().matchAndCreate(I);
  return Op->getKind() != TensorOpKind::Unknown;
}

// Helper function to check if an instruction has a tensor-like access pattern
bool hasTensorAccessPattern(Instruction *I) {
  // Use the tensor operation registry to check for tensor access patterns
  auto Op = TensorOperationRegistry::getInstance().matchAndCreate(I);

  // If the operation is recognized as a tensor operation, it has a tensor access pattern
  if (Op->getKind() != TensorOpKind::Unknown) {
    return true;
  }

  // Check for common tensor access patterns in memory operations
  if (auto *Load = dyn_cast<LoadInst>(I)) {
    return hasMultidimensionalAccess(Load->getPointerOperand());
  }

  if (auto *Store = dyn_cast<StoreInst>(I)) {
    return hasMultidimensionalAccess(Store->getPointerOperand());
  }

  return false;
}

// Helper function to check if a value represents a multidimensional access
bool hasMultidimensionalAccess(Value *V) {
  // Check for GEP instructions with multiple indices
  if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
    // Skip the first index (which is the pointer operand)
    return GEP->getNumIndices() > 1;
  }

  // Check for nested GEP patterns
  if (auto *Load = dyn_cast<LoadInst>(V)) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Load->getPointerOperand())) {
      return GEP->getNumIndices() > 1;
    }
  }

  return false;
}

// Legacy pass implementation
char TensorDataFlowAnalysisWrapperPass::ID = 0;

TensorDataFlowAnalysisWrapperPass::TensorDataFlowAnalysisWrapperPass() : FunctionPass(ID) {}

bool TensorDataFlowAnalysisWrapperPass::runOnFunction(Function &F) {
  // Create a function analysis manager
  FunctionAnalysisManager FAM;

  // Create the analysis pass
  TensorDataFlowAnalysis TDFA;

  // Run the analysis
  Result = TDFA.run(F, FAM);

  // This is an analysis pass, so it doesn't modify the function
  return false;
}

void TensorDataFlowAnalysisWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  // This is an analysis pass, so it doesn't modify the function
  AU.setPreservesAll();
}

// Register the legacy pass
static RegisterPass<TensorDataFlowAnalysisWrapperPass> X("tensor-data-flow-analysis", "Tensor Data Flow Analysis Pass");

// Register the pass with the new pass manager
static PassPluginLibraryInfo getTensorDataFlowAnalysisPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "TensorDataFlowAnalysis", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerAnalysisRegistrationCallback(
        [](FunctionAnalysisManager &FAM) {
          FAM.registerPass([&] { return TensorDataFlowAnalysis(); });
        }
      );
    }
  };
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize the pass when added to the pass pipeline on the command line
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getTensorDataFlowAnalysisPluginInfo();
}

} // namespace tensor
} // namespace llvm
