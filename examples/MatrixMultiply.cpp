#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include "CUDA/TensorKernels.h"
#endif

// Simple matrix class
template <typename T>
class Matrix {
public:
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
  
  T& operator()(int row, int col) {
    return data_[row * cols_ + col];
  }
  
  const T& operator()(int row, int col) const {
    return data_[row * cols_ + col];
  }
  
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  
  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }
  
  size_t size() const { return data_.size(); }
  
private:
  int rows_;
  int cols_;
  std::vector<T> data_;
};

// CPU implementation of matrix multiplication
template <typename T>
void matrixMultiplyCPU(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < B.cols(); ++j) {
      T sum = 0;
      for (int k = 0; k < A.cols(); ++k) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    }
  }
}

// CPU implementation of matrix multiplication with loop tiling
template <typename T>
void matrixMultiplyCPUTiled(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int tileSize) {
  for (int i = 0; i < A.rows(); i += tileSize) {
    for (int j = 0; j < B.cols(); j += tileSize) {
      for (int k = 0; k < A.cols(); k += tileSize) {
        // Process tile
        for (int ii = i; ii < std::min(i + tileSize, A.rows()); ++ii) {
          for (int jj = j; jj < std::min(j + tileSize, B.cols()); ++jj) {
            T sum = 0;
            for (int kk = k; kk < std::min(k + tileSize, A.cols()); ++kk) {
              sum += A(ii, kk) * B(kk, jj);
            }
            C(ii, jj) += sum;
          }
        }
      }
    }
  }
}

#ifdef CUDA_ENABLED
// CUDA implementation of matrix multiplication
template <typename T>
void matrixMultiplyCUDA(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
  // Allocate device memory
  T *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, A.size() * sizeof(T));
  cudaMalloc(&d_B, B.size() * sizeof(T));
  cudaMalloc(&d_C, C.size() * sizeof(T));
  
  // Copy data to device
  cudaMemcpy(d_A, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice);
  
  // Set up dimensions
  int dims[3] = {A.rows(), B.cols(), A.cols()};
  
  // Launch kernel
  llvm::tensor::cuda::launchTensorKernel(
    llvm::tensor::cuda::TensorOpType::MatrixMultiply,
    d_A, d_B, d_C, dims, 0);
  
  // Copy result back to host
  cudaMemcpy(C.data(), d_C, C.size() * sizeof(T), cudaMemcpyDeviceToHost);
  
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
#endif

// Initialize matrix with random values
template <typename T>
void initializeMatrix(Matrix<T>& mat) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(0.0, 1.0);
  
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      mat(i, j) = dis(gen);
    }
  }
}

// Verify matrix multiplication result
template <typename T>
bool verifyResult(const Matrix<T>& expected, const Matrix<T>& actual, T tolerance = 1e-5) {
  if (expected.rows() != actual.rows() || expected.cols() != actual.cols()) {
    return false;
  }
  
  for (int i = 0; i < expected.rows(); ++i) {
    for (int j = 0; j < expected.cols(); ++j) {
      if (std::abs(expected(i, j) - actual(i, j)) > tolerance) {
        return false;
      }
    }
  }
  
  return true;
}

int main(int argc, char** argv) {
  // Parse command line arguments
  int M = 1024;
  int N = 1024;
  int K = 1024;
  int tileSize = 32;
  
  if (argc > 1) M = std::atoi(argv[1]);
  if (argc > 2) N = std::atoi(argv[2]);
  if (argc > 3) K = std::atoi(argv[3]);
  if (argc > 4) tileSize = std::atoi(argv[4]);
  
  std::cout << "Matrix dimensions: A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;
  
  // Create matrices
  Matrix<float> A(M, K);
  Matrix<float> B(K, N);
  Matrix<float> C_cpu(M, N);
  Matrix<float> C_tiled(M, N);
#ifdef CUDA_ENABLED
  Matrix<float> C_cuda(M, N);
#endif
  
  // Initialize matrices
  initializeMatrix(A);
  initializeMatrix(B);
  
  // CPU matrix multiplication
  auto start = std::chrono::high_resolution_clock::now();
  matrixMultiplyCPU(A, B, C_cpu);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "CPU time: " << elapsed.count() << " seconds" << std::endl;
  
  // CPU tiled matrix multiplication
  start = std::chrono::high_resolution_clock::now();
  matrixMultiplyCPUTiled(A, B, C_tiled, tileSize);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "CPU tiled time: " << elapsed.count() << " seconds" << std::endl;
  
  // Verify tiled result
  if (verifyResult(C_cpu, C_tiled)) {
    std::cout << "CPU tiled result is correct" << std::endl;
  } else {
    std::cout << "CPU tiled result is incorrect" << std::endl;
  }
  
#ifdef CUDA_ENABLED
  // CUDA matrix multiplication
  start = std::chrono::high_resolution_clock::now();
  matrixMultiplyCUDA(A, B, C_cuda);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "CUDA time: " << elapsed.count() << " seconds" << std::endl;
  
  // Verify CUDA result
  if (verifyResult(C_cpu, C_cuda)) {
    std::cout << "CUDA result is correct" << std::endl;
  } else {
    std::cout << "CUDA result is incorrect" << std::endl;
  }
#else
  std::cout << "CUDA is not enabled" << std::endl;
#endif
  
  return 0;
}
