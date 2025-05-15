#ifndef LLVM_TENSOR_OPT_AUTO_TUNING_PROFILER_H
#define LLVM_TENSOR_OPT_AUTO_TUNING_PROFILER_H

#include "Analysis/TensorOperationRegistry.h"
#include "AutoTuning/CostModel.h"
#include "llvm/IR/Function.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include <chrono>
#include <memory>
#include <vector>

namespace llvm {
namespace tensor {

/// Class for profiling tensor operations
class Profiler {
public:
  Profiler();
  ~Profiler();
  
  /// Profile a tensor operation with different optimization strategies
  std::vector<std::pair<OptimizationStrategy, double>> profileOperation(
    TensorOperation &Op, const std::vector<OptimizationStrategy> &Strategies);
  
  /// Profile a function with different optimization strategies
  std::vector<std::pair<OptimizationStrategy, double>> profileFunction(
    Function &F, const std::vector<OptimizationStrategy> &Strategies);
  
  /// Get the best optimization strategy based on profiling results
  OptimizationStrategy getBestStrategy(
    const std::vector<std::pair<OptimizationStrategy, double>> &Results) const;
  
private:
  /// LLVM execution engine
  std::unique_ptr<ExecutionEngine> Engine;
  
  /// Apply an optimization strategy to a function
  Function *applyStrategy(Function &F, OptimizationStrategy Strategy);
  
  /// Measure the execution time of a function
  double measureExecutionTime(Function &F, int NumRuns = 10);
};

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_AUTO_TUNING_PROFILER_H
