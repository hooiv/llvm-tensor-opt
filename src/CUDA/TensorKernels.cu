#include "CUDA/TensorKernels.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>

namespace llvm {
namespace tensor {
namespace cuda {

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

// CUDA kernel for element-wise addition
__global__ void elementWiseAddKernel(const float* A, const float* B, float* C, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    C[idx] = A[idx] + B[idx];
  }
}

// CUDA kernel for element-wise multiplication
__global__ void elementWiseMulKernel(const float* A, const float* B, float* C, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    C[idx] = A[idx] * B[idx];
  }
}

// CUDA kernel for reduction (sum)
__global__ void reductionKernel(const float* input, float* output, int size) {
  extern __shared__ float sdata[];
  
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Load input into shared memory
  sdata[tid] = (idx < size) ? input[idx] : 0.0f;
  __syncthreads();
  
  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Write result for this block to output
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

// CUDA kernel for 2D convolution
__global__ void convolutionKernel(const float* input, const float* kernel, float* output,
                                 int inputWidth, int inputHeight, int kernelWidth, int kernelHeight) {
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  
  int outputWidth = inputWidth - kernelWidth + 1;
  int outputHeight = inputHeight - kernelHeight + 1;
  
  if (outRow < outputHeight && outCol < outputWidth) {
    float sum = 0.0f;
    
    for (int kRow = 0; kRow < kernelHeight; ++kRow) {
      for (int kCol = 0; kCol < kernelWidth; ++kCol) {
        int inRow = outRow + kRow;
        int inCol = outCol + kCol;
        sum += input[inRow * inputWidth + inCol] * kernel[kRow * kernelWidth + kCol];
      }
    }
    
    output[outRow * outputWidth + outCol] = sum;
  }
}

// CUDA kernel for fused element-wise operations (add and multiply)
__global__ void fusedElementWiseKernel(const float* A, const float* B, const float* C, float* D, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    // Fused operation: D = (A + B) * C
    D[idx] = (A[idx] + B[idx]) * C[idx];
  }
}

// Launch a CUDA kernel for the specified tensor operation
cudaError_t launchTensorKernel(
  TensorOpType opType,
  const float* inputA,
  const float* inputB,
  float* output,
  const int* dims,
  cudaStream_t stream
) {
  cudaError_t error = cudaSuccess;
  
  switch (opType) {
    case TensorOpType::MatrixMultiply: {
      int M = dims[0];
      int N = dims[1];
      int K = dims[2];
      
      dim3 blockDim(16, 16);
      dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
      
      matrixMultiplyKernel<<<gridDim, blockDim, 0, stream>>>(inputA, inputB, output, M, N, K);
      break;
    }
    
    case TensorOpType::ElementWiseAdd: {
      int size = dims[0];
      
      int blockSize = 256;
      int gridSize = (size + blockSize - 1) / blockSize;
      
      elementWiseAddKernel<<<gridSize, blockSize, 0, stream>>>(inputA, inputB, output, size);
      break;
    }
    
    case TensorOpType::ElementWiseMul: {
      int size = dims[0];
      
      int blockSize = 256;
      int gridSize = (size + blockSize - 1) / blockSize;
      
      elementWiseMulKernel<<<gridSize, blockSize, 0, stream>>>(inputA, inputB, output, size);
      break;
    }
    
    case TensorOpType::Reduction: {
      int size = dims[0];
      
      int blockSize = 256;
      int gridSize = (size + blockSize - 1) / blockSize;
      
      reductionKernel<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(inputA, output, size);
      break;
    }
    
    case TensorOpType::Convolution: {
      int inputWidth = dims[0];
      int inputHeight = dims[1];
      int kernelWidth = dims[2];
      int kernelHeight = dims[3];
      
      int outputWidth = inputWidth - kernelWidth + 1;
      int outputHeight = inputHeight - kernelHeight + 1;
      
      dim3 blockDim(16, 16);
      dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x, (outputHeight + blockDim.y - 1) / blockDim.y);
      
      convolutionKernel<<<gridDim, blockDim, 0, stream>>>(inputA, inputB, output, inputWidth, inputHeight, kernelWidth, kernelHeight);
      break;
    }
    
    default:
      error = cudaErrorInvalidValue;
      break;
  }
  
  return error;
}

// Launch a fused CUDA kernel for multiple tensor operations
cudaError_t launchFusedTensorKernel(
  const TensorOpType* opTypes,
  int numOps,
  const float** inputs,
  float** outputs,
  const int* dims,
  cudaStream_t stream
) {
  cudaError_t error = cudaSuccess;
  
  // For now, only support a specific fusion pattern
  if (numOps == 2 && 
      opTypes[0] == TensorOpType::ElementWiseAdd && 
      opTypes[1] == TensorOpType::ElementWiseMul) {
    
    int size = dims[0];
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    fusedElementWiseKernel<<<gridSize, blockSize, 0, stream>>>(
      inputs[0], inputs[1], inputs[2], outputs[1], size);
  } else {
    error = cudaErrorInvalidValue;
  }
  
  return error;
}

} // namespace cuda
} // namespace tensor
} // namespace llvm

#endif // CUDA_ENABLED
