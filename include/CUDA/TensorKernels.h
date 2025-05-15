#ifndef LLVM_TENSOR_OPT_CUDA_TENSOR_KERNELS_H
#define LLVM_TENSOR_OPT_CUDA_TENSOR_KERNELS_H

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace llvm {
namespace tensor {
namespace cuda {

/// Enum representing different tensor operation types
enum class TensorOpType {
  MatrixMultiply,
  ElementWiseAdd,
  ElementWiseMul,
  Reduction,
  Convolution
};

#ifdef CUDA_ENABLED
/// Launch a CUDA kernel for the specified tensor operation
/// @param opType The type of tensor operation to perform
/// @param inputA First input tensor
/// @param inputB Second input tensor (if applicable)
/// @param output Output tensor
/// @param dims Dimensions of the tensors
/// @param stream CUDA stream to use for kernel launch
cudaError_t launchTensorKernel(
  TensorOpType opType,
  const float* inputA,
  const float* inputB,
  float* output,
  const int* dims,
  cudaStream_t stream = 0
);

/// Launch a fused CUDA kernel for multiple tensor operations
/// @param opTypes Array of tensor operation types to perform
/// @param numOps Number of operations to fuse
/// @param inputs Array of input tensors
/// @param outputs Array of output tensors
/// @param dims Dimensions of the tensors
/// @param stream CUDA stream to use for kernel launch
cudaError_t launchFusedTensorKernel(
  const TensorOpType* opTypes,
  int numOps,
  const float** inputs,
  float** outputs,
  const int* dims,
  cudaStream_t stream = 0
);
#endif

} // namespace cuda
} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_CUDA_TENSOR_KERNELS_H
