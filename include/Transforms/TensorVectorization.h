#ifndef LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_VECTORIZATION_H
#define LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_VECTORIZATION_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {
namespace tensor {

/// Pass to vectorize tensor operations for SIMD execution on GPUs
/// This pass identifies tensor operations that can benefit from
/// vectorization and transforms them to use vector instructions
class TensorVectorizationPass : public PassInfoMixin<TensorVectorizationPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  
  // Legacy pass interface
  static char ID;
};

/// Creates a pass to perform tensor operation vectorization
std::unique_ptr<FunctionPass> createTensorVectorizationPass();

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_VECTORIZATION_H
