#ifndef LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_FUSION_H
#define LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_FUSION_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {
namespace tensor {

/// Pass to fuse tensor operations for better GPU execution
/// This pass identifies tensor operations that can be fused together
/// to reduce memory traffic and improve GPU utilization
class TensorFusionPass : public PassInfoMixin<TensorFusionPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  
  // Legacy pass interface
  static char ID;
};

/// Creates a pass to perform tensor operation fusion
std::unique_ptr<FunctionPass> createTensorFusionPass();

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_TRANSFORMS_TENSOR_FUSION_H
