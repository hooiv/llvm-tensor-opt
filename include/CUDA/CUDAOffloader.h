#ifndef LLVM_TENSOR_OPT_CUDA_OFFLOADER_H
#define LLVM_TENSOR_OPT_CUDA_OFFLOADER_H

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"

namespace llvm {
namespace tensor {
namespace cuda {

/// Class to handle offloading of tensor operations to CUDA
/// This class transforms LLVM IR to offload tensor operations to CUDA
class CUDAOffloader {
public:
  CUDAOffloader();
  ~CUDAOffloader();

  /// Offload tensor operations in the given function to CUDA
  /// @param F The function containing tensor operations to offload
  /// @return True if any operations were offloaded, false otherwise
  bool offloadFunction(Function &F);

  /// Offload tensor operations in the given module to CUDA
  /// @param M The module containing tensor operations to offload
  /// @return True if any operations were offloaded, false otherwise
  bool offloadModule(Module &M);

private:
  /// Identify tensor operations in the given function
  /// @param F The function to analyze
  /// @return A vector of instructions representing tensor operations
  std::vector<Instruction*> identifyTensorOperations(Function &F);

  /// Create CUDA kernel launch for the given tensor operation
  /// @param I The instruction representing a tensor operation
  /// @return True if the operation was successfully offloaded, false otherwise
  bool createCUDAKernelLaunch(Instruction &I);
};

} // namespace cuda
} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_CUDA_OFFLOADER_H
