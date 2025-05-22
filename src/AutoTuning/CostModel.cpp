#include "AutoTuning/CostModel.h"
#include "Analysis/TensorOperationRegistry.h"
#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorAccessPatternAnalysis.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <vector>
#include <memory>
#include <algorithm>

#define DEBUG_TYPE "cost-model"

// Use explicit namespace references instead of 'using namespace'
// to avoid namespace conflicts

// SimpleCostModel implementation
llvm::tensor::SimpleCostModel::SimpleCostModel() {
  // Initialize cost factors for different tensor operations
  OpCostFactors[llvm::tensor::TensorOpKind::MatrixMultiply] = 2.0;
  OpCostFactors[llvm::tensor::TensorOpKind::Convolution] = 3.0;
  OpCostFactors[llvm::tensor::TensorOpKind::ElementWiseAdd] = 1.0;
  OpCostFactors[llvm::tensor::TensorOpKind::ElementWiseMul] = 1.0;
  OpCostFactors[llvm::tensor::TensorOpKind::Reduction] = 1.5;
  OpCostFactors[llvm::tensor::TensorOpKind::Transpose] = 1.2;
  OpCostFactors[llvm::tensor::TensorOpKind::Reshape] = 0.5;
  OpCostFactors[llvm::tensor::TensorOpKind::Concat] = 1.0;
  OpCostFactors[llvm::tensor::TensorOpKind::Split] = 1.0;
  OpCostFactors[llvm::tensor::TensorOpKind::Attention] = 2.5;
  OpCostFactors[llvm::tensor::TensorOpKind::LayerNorm] = 1.5;
  OpCostFactors[llvm::tensor::TensorOpKind::Softmax] = 1.2;
  OpCostFactors[llvm::tensor::TensorOpKind::Pooling] = 1.5;
  OpCostFactors[llvm::tensor::TensorOpKind::BatchNorm] = 1.5;
  OpCostFactors[llvm::tensor::TensorOpKind::Activation] = 0.8;
  OpCostFactors[llvm::tensor::TensorOpKind::TensorContraction] = 2.0;
  OpCostFactors[llvm::tensor::TensorOpKind::Unknown] = 1.0;

  // Initialize cost reduction factors for different optimization strategies
  StrategyCostFactors[llvm::tensor::OptimizationStrategy::None] = 1.0;
  StrategyCostFactors[llvm::tensor::OptimizationStrategy::Fusion] = 0.8;
  StrategyCostFactors[llvm::tensor::OptimizationStrategy::Vectorization] = 0.7;
  StrategyCostFactors[llvm::tensor::OptimizationStrategy::Parallelization] = 0.5;
  StrategyCostFactors[llvm::tensor::OptimizationStrategy::FusionAndVectorization] = 0.6;
  StrategyCostFactors[llvm::tensor::OptimizationStrategy::FusionAndParallelization] = 0.4;
  StrategyCostFactors[llvm::tensor::OptimizationStrategy::VectorizationAndParallelization] = 0.3;
  StrategyCostFactors[llvm::tensor::OptimizationStrategy::All] = 0.2;
}

double llvm::tensor::SimpleCostModel::estimateCost(const llvm::tensor::TensorOperation &Op) const {
  // Get the cost factor for this operation kind
  auto It = OpCostFactors.find(Op.getKind());
  double CostFactor = It != OpCostFactors.end() ? It->second : 1.0;

  // Get the dimensions of the operation
  auto Dims = Op.getDimensions();

  // Calculate the cost based on the dimensions
  double Cost = CostFactor;
  for (auto Dim : Dims) {
    if (Dim > 0) {
      Cost *= Dim;
    }
  }

  return Cost;
}

double llvm::tensor::SimpleCostModel::estimateCost(const llvm::Function &F) const {
  double TotalCost = 0.0;

  // Iterate over all instructions in the function
  for (const auto &BB : F) {
    for (const auto &I : BB) {
      // Create a tensor operation from the instruction
      auto Op = createTensorOperation(const_cast<Instruction *>(&I));

      // Add the cost of this operation to the total cost
      TotalCost += estimateCost(*Op);
    }
  }

  return TotalCost;
}

double llvm::tensor::SimpleCostModel::estimateCost(const llvm::tensor::TensorOperation &Op, llvm::tensor::OptimizationStrategy Strategy) const {
  // Get the base cost of the operation
  double BaseCost = estimateCost(Op);

  // Apply the cost reduction factor for the optimization strategy
  auto It = StrategyCostFactors.find(Strategy);
  double ReductionFactor = It != StrategyCostFactors.end() ? It->second : 1.0;

  return BaseCost * ReductionFactor;
}

double llvm::tensor::SimpleCostModel::estimateCost(const llvm::Function &F, llvm::tensor::OptimizationStrategy Strategy) const {
  // Get the base cost of the function
  double BaseCost = estimateCost(F);

  // Apply the cost reduction factor for the optimization strategy
  auto It = StrategyCostFactors.find(Strategy);
  double ReductionFactor = It != StrategyCostFactors.end() ? It->second : 1.0;

  return BaseCost * ReductionFactor;
}

llvm::tensor::OptimizationStrategy llvm::tensor::SimpleCostModel::getBestStrategy(const llvm::tensor::TensorOperation &Op) const {
  // Try all optimization strategies and pick the one with the lowest cost
  llvm::tensor::OptimizationStrategy BestStrategy = llvm::tensor::OptimizationStrategy::None;
  double BestCost = estimateCost(Op, BestStrategy);

  for (auto It = StrategyCostFactors.begin(); It != StrategyCostFactors.end(); ++It) {
    llvm::tensor::OptimizationStrategy Strategy = It->first;
    double Cost = estimateCost(Op, Strategy);

    if (Cost < BestCost) {
      BestCost = Cost;
      BestStrategy = Strategy;
    }
  }

  return BestStrategy;
}

llvm::tensor::OptimizationStrategy llvm::tensor::SimpleCostModel::getBestStrategy(const llvm::Function &F) const {
  // Try all optimization strategies and pick the one with the lowest cost
  llvm::tensor::OptimizationStrategy BestStrategy = llvm::tensor::OptimizationStrategy::None;
  double BestCost = estimateCost(F, BestStrategy);

  for (auto It = StrategyCostFactors.begin(); It != StrategyCostFactors.end(); ++It) {
    llvm::tensor::OptimizationStrategy Strategy = It->first;
    double Cost = estimateCost(F, Strategy);

    if (Cost < BestCost) {
      BestCost = Cost;
      BestStrategy = Strategy;
    }
  }

  return BestStrategy;
}

// MLCostModel implementation
llvm::tensor::MLCostModel::MLCostModel() {
  // Initialize model weights
  Weights = {1.0, 0.5, 0.3, 0.2, 0.1};
}

double llvm::tensor::MLCostModel::estimateCost(const llvm::tensor::TensorOperation &Op) const {
  // Extract features from the operation
  auto Features = extractFeatures(Op);

  // Predict the cost using the ML model
  return predict(Features);
}

double llvm::tensor::MLCostModel::estimateCost(const llvm::Function &F) const {
  // Extract features from the function
  auto Features = extractFeatures(F);

  // Predict the cost using the ML model
  return predict(Features);
}

double llvm::tensor::MLCostModel::estimateCost(const llvm::tensor::TensorOperation &Op, llvm::tensor::OptimizationStrategy Strategy) const {
  // Extract features from the operation
  auto Features = extractFeatures(Op);

  // Add features for the optimization strategy
  Features.push_back(static_cast<double>(Strategy));

  // Predict the cost using the ML model
  return predict(Features);
}

double llvm::tensor::MLCostModel::estimateCost(const llvm::Function &F, llvm::tensor::OptimizationStrategy Strategy) const {
  // Extract features from the function
  auto Features = extractFeatures(F);

  // Add features for the optimization strategy
  Features.push_back(static_cast<double>(Strategy));

  // Predict the cost using the ML model
  return predict(Features);
}

llvm::tensor::OptimizationStrategy llvm::tensor::MLCostModel::getBestStrategy(const llvm::tensor::TensorOperation &Op) const {
  // Try all optimization strategies and pick the one with the lowest cost
  llvm::tensor::OptimizationStrategy BestStrategy = llvm::tensor::OptimizationStrategy::None;
  double BestCost = estimateCost(Op, BestStrategy);

  for (int i = 0; i <= static_cast<int>(llvm::tensor::OptimizationStrategy::All); ++i) {
    llvm::tensor::OptimizationStrategy Strategy = static_cast<llvm::tensor::OptimizationStrategy>(i);
    double Cost = estimateCost(Op, Strategy);

    if (Cost < BestCost) {
      BestCost = Cost;
      BestStrategy = Strategy;
    }
  }

  return BestStrategy;
}

llvm::tensor::OptimizationStrategy llvm::tensor::MLCostModel::getBestStrategy(const llvm::Function &F) const {
  // Try all optimization strategies and pick the one with the lowest cost
  llvm::tensor::OptimizationStrategy BestStrategy = llvm::tensor::OptimizationStrategy::None;
  double BestCost = estimateCost(F, BestStrategy);

  for (int i = 0; i <= static_cast<int>(llvm::tensor::OptimizationStrategy::All); ++i) {
    llvm::tensor::OptimizationStrategy Strategy = static_cast<llvm::tensor::OptimizationStrategy>(i);
    double Cost = estimateCost(F, Strategy);

    if (Cost < BestCost) {
      BestCost = Cost;
      BestStrategy = Strategy;
    }
  }

  return BestStrategy;
}

void llvm::tensor::MLCostModel::train(const std::vector<std::pair<llvm::tensor::TensorOperation *, double>> &TrainingData) {
  // Simple linear regression training
  // This is a placeholder implementation

  // Extract features and targets from the training data
  std::vector<std::vector<double>> Features;
  std::vector<double> Targets;

  for (const auto &Data : TrainingData) {
    Features.push_back(extractFeatures(*Data.first));
    Targets.push_back(Data.second);
  }

  // Train the model
  // This is a simplified implementation
  Weights.resize(Features[0].size());
  for (size_t i = 0; i < Weights.size(); ++i) {
    Weights[i] = 1.0 / Weights.size();
  }
}

std::vector<double> llvm::tensor::MLCostModel::extractFeatures(const llvm::tensor::TensorOperation &Op) const {
  // Extract features from the operation
  std::vector<double> Features;

  // Add the operation kind as a feature
  Features.push_back(static_cast<double>(Op.getKind()));

  // Add the dimensions as features
  auto Dims = Op.getDimensions();
  for (auto Dim : Dims) {
    Features.push_back(static_cast<double>(Dim));
  }

  // Add other features
  Features.push_back(Op.canVectorize() ? 1.0 : 0.0);
  Features.push_back(Op.canParallelize() ? 1.0 : 0.0);

  return Features;
}

std::vector<double> llvm::tensor::MLCostModel::extractFeatures(const llvm::Function &F) const {
  // Extract features from the function
  std::vector<double> Features;

  // Count the number of each type of tensor operation
  std::map<llvm::tensor::TensorOpKind, int> OpCounts;

  // Iterate over all instructions in the function
  for (const auto &BB : F) {
    for (const auto &I : BB) {
      // Create a tensor operation from the instruction
      auto Op = createTensorOperation(const_cast<Instruction *>(&I));

      // Increment the count for this operation kind
      ++OpCounts[Op->getKind()];
    }
  }

  // Add the operation counts as features
  for (int i = 0; i <= static_cast<int>(llvm::tensor::TensorOpKind::TensorContraction); ++i) {
    llvm::tensor::TensorOpKind Kind = static_cast<llvm::tensor::TensorOpKind>(i);
    Features.push_back(static_cast<double>(OpCounts[Kind]));
  }

  // Add other features
  Features.push_back(static_cast<double>(F.size()));
  // Count basic blocks directly instead of using private getBasicBlockList()
  unsigned NumBasicBlocks = 0;
  for (const auto &BB : F) {
    NumBasicBlocks++;
  }
  Features.push_back(static_cast<double>(NumBasicBlocks));

  return Features;
}

double llvm::tensor::MLCostModel::predict(const std::vector<double> &Features) const {
  // Simple linear model prediction
  double Prediction = 0.0;

  for (size_t i = 0; i < std::min(Features.size(), Weights.size()); ++i) {
    Prediction += Features[i] * Weights[i];
  }

  return Prediction;
}

// Factory function to create a cost model
std::unique_ptr<llvm::tensor::CostModel> llvm::tensor::createCostModel(bool UseML) {
  if (UseML) {
    return std::make_unique<MLCostModel>();
  } else {
    return std::make_unique<SimpleCostModel>();
  }
}
