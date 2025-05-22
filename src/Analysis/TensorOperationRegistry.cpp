#include "Analysis/TensorOperationRegistry.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-operation-registry"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm {
namespace tensor {

StringRef TensorOperation::getName() const {
  switch (Kind) {
  case TensorOpKind::MatrixMultiply:
    return "MatrixMultiply";
  case TensorOpKind::Convolution:
    return "Convolution";
  case TensorOpKind::ElementWiseAdd:
    return "ElementWiseAdd";
  case TensorOpKind::ElementWiseMul:
    return "ElementWiseMul";
  case TensorOpKind::Reduction:
    return "Reduction";
  case TensorOpKind::Transpose:
    return "Transpose";
  case TensorOpKind::Reshape:
    return "Reshape";
  case TensorOpKind::Concat:
    return "Concat";
  case TensorOpKind::Split:
    return "Split";
  case TensorOpKind::Attention:
    return "Attention";
  case TensorOpKind::LayerNorm:
    return "LayerNorm";
  case TensorOpKind::Softmax:
    return "Softmax";
  case TensorOpKind::Pooling:
    return "Pooling";
  case TensorOpKind::BatchNorm:
    return "BatchNorm";
  case TensorOpKind::Activation:
    return "Activation";
  case TensorOpKind::TensorContraction:
    return "TensorContraction";
  case TensorOpKind::Unknown:
  default:
    return "Unknown";
  }
}

bool TensorOperation::canFuseWith(const TensorOperation &Other) const {
  // Default implementation: operations of the same kind can be fused
  return Kind == Other.Kind;
}

bool TensorOperation::canVectorize() const {
  // Default implementation: most operations can be vectorized
  return Kind != TensorOpKind::Unknown;
}

bool TensorOperation::canParallelize() const {
  // Default implementation: most operations can be parallelized
  return Kind != TensorOpKind::Unknown;
}

std::vector<Value *> TensorOperation::getOperands() const {
  std::vector<Value *> Operands;
  for (Use &U : Inst->operands()) {
    Operands.push_back(U.get());
  }
  return Operands;
}

std::vector<Value *> TensorOperation::getResults() const {
  std::vector<Value *> Results;
  Results.push_back(Inst);
  return Results;
}

std::vector<int64_t> TensorOperation::getDimensions() const {
  // Default implementation: no dimensions
  return {};
}

std::string TensorOperation::toString() const {
  return getName().str();
}

// Matrix Multiply implementation
bool MatrixMultiplyOp::canFuseWith(const TensorOperation &Other) const {
  // Matrix multiply can be fused with element-wise operations
  return Other.getKind() == TensorOpKind::ElementWiseAdd ||
         Other.getKind() == TensorOpKind::ElementWiseMul ||
         Other.getKind() == TensorOpKind::Activation;
}

std::vector<Value *> MatrixMultiplyOp::getOperands() const {
  // For a matrix multiply, we expect at least 2 operands (A and B)
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (Operands.size() >= 2) {
    return {Operands[0], Operands[1]};
  }
  return Operands;
}

std::vector<Value *> MatrixMultiplyOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> MatrixMultiplyOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1, -1, -1}; // M, N, K
}

std::string MatrixMultiplyOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 3 && Dims[0] != -1 && Dims[1] != -1 && Dims[2] != -1) {
    return "MatrixMultiply(" + std::to_string(Dims[0]) + "x" +
           std::to_string(Dims[1]) + "x" + std::to_string(Dims[2]) + ")";
  }
  return "MatrixMultiply";
}

// Convolution implementation
bool ConvolutionOp::canFuseWith(const TensorOperation &Other) const {
  // Convolution can be fused with element-wise operations and activations
  return Other.getKind() == TensorOpKind::ElementWiseAdd ||
         Other.getKind() == TensorOpKind::ElementWiseMul ||
         Other.getKind() == TensorOpKind::Activation ||
         Other.getKind() == TensorOpKind::BatchNorm;
}

std::vector<Value *> ConvolutionOp::getOperands() const {
  // For a convolution, we expect at least 2 operands (input and kernel)
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (Operands.size() >= 2) {
    return {Operands[0], Operands[1]};
  }
  return Operands;
}

std::vector<Value *> ConvolutionOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> ConvolutionOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1, -1, -1, -1}; // N, C, H, W
}

std::string ConvolutionOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 4 && Dims[0] != -1 && Dims[1] != -1 && Dims[2] != -1 && Dims[3] != -1) {
    return "Convolution(" + std::to_string(Dims[0]) + "x" +
           std::to_string(Dims[1]) + "x" + std::to_string(Dims[2]) + "x" +
           std::to_string(Dims[3]) + ")";
  }
  return "Convolution";
}

// Factory function to create a tensor operation from an instruction
std::unique_ptr<TensorOperation> createTensorOperation(Instruction *Inst) {
  return TensorOperationRegistry::getInstance().matchAndCreate(Inst);
}

// Singleton instance of the registry
TensorOperationRegistry &TensorOperationRegistry::getInstance() {
  static TensorOperationRegistry Instance;
  return Instance;
}

// Register a pattern for a tensor operation
void TensorOperationRegistry::registerPattern(PatternMatcherTy Matcher,
                                             TensorOpCreatorTy Creator,
                                             StringRef Name) {
  Patterns.emplace_back(std::move(Matcher), std::move(Creator), Name);
}

// Match an instruction to a tensor operation
std::unique_ptr<TensorOperation> TensorOperationRegistry::matchAndCreate(Instruction *Inst) const {
  for (const auto &Pattern : Patterns) {
    if (Pattern.Matcher(Inst)) {
      LLVM_DEBUG(dbgs() << "Matched tensor operation: " << Pattern.Name << "\n");
      return Pattern.Creator(Inst);
    }
  }

  // No match found, return a generic tensor operation
  return std::make_unique<TensorOperation>(TensorOpKind::Unknown, Inst);
}

// Register built-in tensor operation patterns
namespace {

// Helper function to check if a function name contains a substring
bool functionNameContains(const CallInst *Call, StringRef Substring) {
  if (const Function *F = Call->getCalledFunction()) {
    return F->getName().contains(Substring);
  }
  return false;
}

// Pattern matcher for matrix multiply operations
bool isMatrixMultiply(Instruction *Inst) {
  // Check for direct calls to matrix multiply functions
  if (auto *Call = dyn_cast<CallInst>(Inst)) {
    if (functionNameContains(Call, "matmul") ||
        functionNameContains(Call, "gemm") ||
        functionNameContains(Call, "matrix_multiply") ||
        functionNameContains(Call, "dot")) {
      return true;
    }
  }

  // Check for nested loops with multiply-accumulate pattern
  // This is a simplified check and would need to be more sophisticated in practice
  if (auto *Store = dyn_cast<StoreInst>(Inst)) {
    Value *StoredVal = Store->getValueOperand();
    if (auto *Add = dyn_cast<BinaryOperator>(StoredVal)) {
      if (Add->getOpcode() == Instruction::FAdd || Add->getOpcode() == Instruction::Add) {
        Value *LHS = Add->getOperand(0);
        Value *RHS = Add->getOperand(1);
        if (auto *Mul = dyn_cast<BinaryOperator>(RHS)) {
          if (Mul->getOpcode() == Instruction::FMul || Mul->getOpcode() == Instruction::Mul) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

// Pattern matcher for convolution operations
bool isConvolution(Instruction *Inst) {
  // Check for direct calls to convolution functions
  if (auto *Call = dyn_cast<CallInst>(Inst)) {
    if (functionNameContains(Call, "conv") ||
        functionNameContains(Call, "convolution")) {
      return true;
    }
  }

  // More sophisticated pattern matching would be needed for detecting
  // convolution patterns in the IR

  return false;
}

// Register built-in patterns
static RegisterTensorOperation<MatrixMultiplyOp> RegisterMatMul("MatrixMultiply", isMatrixMultiply);
static RegisterTensorOperation<ConvolutionOp> RegisterConv("Convolution", isConvolution);

} // anonymous namespace

} // namespace tensor
} // namespace llvm
