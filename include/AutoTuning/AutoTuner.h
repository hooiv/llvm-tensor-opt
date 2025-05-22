#ifndef LLVM_TENSOR_OPT_AUTO_TUNING_AUTO_TUNER_H
#define LLVM_TENSOR_OPT_AUTO_TUNING_AUTO_TUNER_H

#include "AutoTuning/CostModel.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include <memory>

namespace llvm {
namespace tensor {

/// Class for auto-tuning tensor operations
class AutoTuner {
public:
  AutoTuner(bool UseML = false);

  /// Tune a function and apply the best optimization strategy
  bool tune(Function &F, FunctionAnalysisManager &AM);

  /// Get the best optimization strategy for a function
  OptimizationStrategy getBestStrategy(const Function &F) const;

  /// Get the cost model
  const CostModel &getCostModel() const { return *CostModelImpl; }

  /// Apply an optimization strategy to a function
  bool applyStrategy(Function &F, FunctionAnalysisManager &AM, OptimizationStrategy Strategy);

private:
  /// Cost model implementation
  std::unique_ptr<CostModel> CostModelImpl;
};

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_AUTO_TUNING_AUTO_TUNER_H
