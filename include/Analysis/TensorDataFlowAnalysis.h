#ifndef LLVM_TENSOR_OPT_ANALYSIS_TENSOR_DATA_FLOW_ANALYSIS_H
#define LLVM_TENSOR_OPT_ANALYSIS_TENSOR_DATA_FLOW_ANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
namespace tensor {

/// Analysis pass to identify tensor operations and their data flow
/// This pass analyzes the data flow between tensor operations to
/// identify optimization opportunities
class TensorDataFlowAnalysis : public AnalysisInfoMixin<TensorDataFlowAnalysis> {
public:
  using Result = DenseMap<Instruction*, std::vector<Instruction*>>;
  
  Result run(Function &F, FunctionAnalysisManager &AM);
  
  // Unique ID for analysis pass
  static AnalysisKey Key;
};

/// Legacy pass interface for tensor data flow analysis
class TensorDataFlowAnalysisWrapperPass : public FunctionPass {
  TensorDataFlowAnalysis::Result Result;

public:
  static char ID;
  TensorDataFlowAnalysisWrapperPass();

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  
  const TensorDataFlowAnalysis::Result &getResult() const { return Result; }
};

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_ANALYSIS_TENSOR_DATA_FLOW_ANALYSIS_H
