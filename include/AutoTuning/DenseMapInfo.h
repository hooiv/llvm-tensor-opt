#ifndef LLVM_TENSOR_OPT_AUTO_TUNING_DENSE_MAP_INFO_H
#define LLVM_TENSOR_OPT_AUTO_TUNING_DENSE_MAP_INFO_H

#include "Analysis/TensorOperationRegistry.h"
#include "llvm/ADT/DenseMapInfo.h"

// Forward declaration of OptimizationStrategy enum
namespace llvm {
namespace tensor {
enum class OptimizationStrategy {
  None,
  Fusion,
  Vectorization,
  Parallelization,
  FusionAndVectorization,
  FusionAndParallelization,
  VectorizationAndParallelization,
  All
};
} // end namespace tensor

// Specialization of DenseMapInfo for TensorOpKind
template<>
struct DenseMapInfo<tensor::TensorOpKind> {
  static inline tensor::TensorOpKind getEmptyKey() {
    return static_cast<tensor::TensorOpKind>(-1);
  }

  static inline tensor::TensorOpKind getTombstoneKey() {
    return static_cast<tensor::TensorOpKind>(-2);
  }

  static unsigned getHashValue(const tensor::TensorOpKind &Val) {
    return static_cast<unsigned>(Val);
  }

  static bool isEqual(const tensor::TensorOpKind &LHS, const tensor::TensorOpKind &RHS) {
    return LHS == RHS;
  }
};

// Specialization of DenseMapInfo for OptimizationStrategy
template<>
struct DenseMapInfo<tensor::OptimizationStrategy> {
  static inline tensor::OptimizationStrategy getEmptyKey() {
    return static_cast<tensor::OptimizationStrategy>(-1);
  }

  static inline tensor::OptimizationStrategy getTombstoneKey() {
    return static_cast<tensor::OptimizationStrategy>(-2);
  }

  static unsigned getHashValue(const tensor::OptimizationStrategy &Val) {
    return static_cast<unsigned>(Val);
  }

  static bool isEqual(const tensor::OptimizationStrategy &LHS, const tensor::OptimizationStrategy &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_TENSOR_OPT_AUTO_TUNING_DENSE_MAP_INFO_H
