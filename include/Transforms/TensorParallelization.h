#ifndef LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_PARALLELIZATION_H
#define LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_PARALLELIZATION_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {
namespace tensor {

/// Pass to parallelize tensor operations for GPU execution
/// This pass identifies tensor operations that can be parallelized
/// and transforms them to exploit GPU parallelism
class TensorParallelizationPass : public PassInfoMixin<TensorParallelizationPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  
  // Legacy pass interface
  static char ID;
};

/// Creates a pass to perform tensor operation parallelization
std::unique_ptr<FunctionPass> createTensorParallelizationPass();

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_PARALLELIZATION_H
