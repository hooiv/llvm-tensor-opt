#ifndef LLVM_TENSOR_OPT_AUTO_TUNING_COST_MODEL_H
#define LLVM_TENSOR_OPT_AUTO_TUNING_COST_MODEL_H

#include "Analysis/TensorOperationRegistry.h"
#include "AutoTuning/DenseMapInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>
#include <memory>

namespace llvm {
namespace tensor {

/// Enum representing different optimization strategies
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

/// Class representing a cost model for tensor operations
class CostModel {
public:
  virtual ~CostModel() = default;

  /// Estimate the cost of a tensor operation
  virtual double estimateCost(const TensorOperation &Op) const = 0;

  /// Estimate the cost of a function
  virtual double estimateCost(const Function &F) const = 0;

  /// Estimate the cost of a tensor operation with a specific optimization strategy
  virtual double estimateCost(const TensorOperation &Op, OptimizationStrategy Strategy) const = 0;

  /// Estimate the cost of a function with a specific optimization strategy
  virtual double estimateCost(const Function &F, OptimizationStrategy Strategy) const = 0;

  /// Get the best optimization strategy for a tensor operation
  virtual OptimizationStrategy getBestStrategy(const TensorOperation &Op) const = 0;

  /// Get the best optimization strategy for a function
  virtual OptimizationStrategy getBestStrategy(const Function &F) const = 0;
};

/// Class representing a simple cost model based on operation count
class SimpleCostModel : public CostModel {
public:
  SimpleCostModel();

  double estimateCost(const TensorOperation &Op) const override;
  double estimateCost(const Function &F) const override;
  double estimateCost(const TensorOperation &Op, OptimizationStrategy Strategy) const override;
  double estimateCost(const Function &F, OptimizationStrategy Strategy) const override;
  OptimizationStrategy getBestStrategy(const TensorOperation &Op) const override;
  OptimizationStrategy getBestStrategy(const Function &F) const override;

private:
  /// Cost factors for different tensor operations
  DenseMap<TensorOpKind, double> OpCostFactors;

  /// Cost reduction factors for different optimization strategies
  DenseMap<OptimizationStrategy, double> StrategyCostFactors;
};

/// Class representing a machine learning-based cost model
class MLCostModel : public CostModel {
public:
  MLCostModel();

  double estimateCost(const TensorOperation &Op) const override;
  double estimateCost(const Function &F) const override;
  double estimateCost(const TensorOperation &Op, OptimizationStrategy Strategy) const override;
  double estimateCost(const Function &F, OptimizationStrategy Strategy) const override;
  OptimizationStrategy getBestStrategy(const TensorOperation &Op) const override;
  OptimizationStrategy getBestStrategy(const Function &F) const override;

  /// Train the cost model with new data
  void train(const std::vector<std::pair<TensorOperation *, double>> &TrainingData);

private:
  /// Feature extraction for a tensor operation
  std::vector<double> extractFeatures(const TensorOperation &Op) const;

  /// Feature extraction for a function
  std::vector<double> extractFeatures(const Function &F) const;

  /// Predict the cost using the ML model
  double predict(const std::vector<double> &Features) const;

  /// Model weights
  std::vector<double> Weights;
};

/// Factory function to create a cost model
std::unique_ptr<CostModel> createCostModel(bool UseML = false);

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_AUTO_TUNING_COST_MODEL_H
