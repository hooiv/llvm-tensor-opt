#ifndef LLVM_TENSOR_OPT_ANALYSIS_TENSOR_ACCESS_PATTERN_ANALYSIS_H
#define LLVM_TENSOR_OPT_ANALYSIS_TENSOR_ACCESS_PATTERN_ANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
namespace tensor {

/// Enum representing different tensor access patterns
enum class AccessPattern {
  Unknown,
  Sequential,
  Strided,
  Random,
  Broadcast
};

/// Analysis pass to identify tensor access patterns
/// This pass analyzes memory access patterns in tensor operations
/// to identify optimization opportunities for GPU execution
class TensorAccessPatternAnalysis : public AnalysisInfoMixin<TensorAccessPatternAnalysis> {
public:
  using Result = DenseMap<Instruction*, AccessPattern>;
  
  Result run(Function &F, FunctionAnalysisManager &AM);
  
  // Unique ID for analysis pass
  static AnalysisKey Key;
};

/// Legacy pass interface for tensor access pattern analysis
class TensorAccessPatternAnalysisWrapperPass : public FunctionPass {
  TensorAccessPatternAnalysis::Result Result;

public:
  static char ID;
  TensorAccessPatternAnalysisWrapperPass();

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  
  const TensorAccessPatternAnalysis::Result &getResult() const { return Result; }
};

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_ANALYSIS_TENSOR_ACCESS_PATTERN_ANALYSIS_H
