#include "Analysis/TensorOperations.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-operations"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm {
namespace tensor {

// ElementWiseAddOp implementation
bool ElementWiseAddOp::canFuseWith(const TensorOperation &Other) const {
  // Element-wise add can be fused with most operations
  return Other.getKind() != TensorOpKind::Unknown;
}

std::vector<Value *> ElementWiseAddOp::getOperands() const {
  // For an element-wise add, we expect 2 operands
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (Operands.size() >= 2) {
    return {Operands[0], Operands[1]};
  }
  return Operands;
}

std::vector<Value *> ElementWiseAddOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> ElementWiseAddOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1}; // Size
}

std::string ElementWiseAddOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 1 && Dims[0] != -1) {
    return "ElementWiseAdd(" + std::to_string(Dims[0]) + ")";
  }
  return "ElementWiseAdd";
}

// ElementWiseMulOp implementation
bool ElementWiseMulOp::canFuseWith(const TensorOperation &Other) const {
  // Element-wise mul can be fused with most operations
  return Other.getKind() != TensorOpKind::Unknown;
}

std::vector<Value *> ElementWiseMulOp::getOperands() const {
  // For an element-wise mul, we expect 2 operands
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (Operands.size() >= 2) {
    return {Operands[0], Operands[1]};
  }
  return Operands;
}

std::vector<Value *> ElementWiseMulOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> ElementWiseMulOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1}; // Size
}

std::string ElementWiseMulOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 1 && Dims[0] != -1) {
    return "ElementWiseMul(" + std::to_string(Dims[0]) + ")";
  }
  return "ElementWiseMul";
}

// ReductionOp implementation
bool ReductionOp::canFuseWith(const TensorOperation &Other) const {
  // Reduction can be fused with element-wise operations
  return Other.getKind() == TensorOpKind::ElementWiseAdd ||
         Other.getKind() == TensorOpKind::ElementWiseMul;
}

std::vector<Value *> ReductionOp::getOperands() const {
  // For a reduction, we expect 1 operand
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (!Operands.empty()) {
    return {Operands[0]};
  }
  return Operands;
}

std::vector<Value *> ReductionOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> ReductionOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1, -1}; // Input size, output size
}

std::string ReductionOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 2 && Dims[0] != -1 && Dims[1] != -1) {
    return "Reduction(" + std::to_string(Dims[0]) + " -> " +
           std::to_string(Dims[1]) + ")";
  }
  return "Reduction";
}

// TransposeOp implementation
bool TransposeOp::canFuseWith(const TensorOperation &Other) const {
  // Transpose can be fused with element-wise operations
  return Other.getKind() == TensorOpKind::ElementWiseAdd ||
         Other.getKind() == TensorOpKind::ElementWiseMul;
}

std::vector<Value *> TransposeOp::getOperands() const {
  // For a transpose, we expect 1 operand
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (!Operands.empty()) {
    return {Operands[0]};
  }
  return Operands;
}

std::vector<Value *> TransposeOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> TransposeOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1, -1}; // Rows, columns
}

std::string TransposeOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 2 && Dims[0] != -1 && Dims[1] != -1) {
    return "Transpose(" + std::to_string(Dims[0]) + "x" +
           std::to_string(Dims[1]) + ")";
  }
  return "Transpose";
}

// ReshapeOp implementation
bool ReshapeOp::canFuseWith(const TensorOperation &Other) const {
  // Reshape generally cannot be fused with other operations
  return false;
}

std::vector<Value *> ReshapeOp::getOperands() const {
  // For a reshape, we expect 1 operand
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (!Operands.empty()) {
    return {Operands[0]};
  }
  return Operands;
}

std::vector<Value *> ReshapeOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> ReshapeOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1, -1}; // Input size, output size
}

std::string ReshapeOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 2 && Dims[0] != -1 && Dims[1] != -1) {
    return "Reshape(" + std::to_string(Dims[0]) + " -> " +
           std::to_string(Dims[1]) + ")";
  }
  return "Reshape";
}

// AttentionOp implementation
bool AttentionOp::canFuseWith(const TensorOperation &Other) const {
  // Attention can be fused with softmax and layer normalization
  return Other.getKind() == TensorOpKind::Softmax ||
         Other.getKind() == TensorOpKind::LayerNorm;
}

std::vector<Value *> AttentionOp::getOperands() const {
  // For an attention operation, we expect 3 operands (query, key, value)
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (Operands.size() >= 3) {
    return {Operands[0], Operands[1], Operands[2]};
  }
  return Operands;
}

std::vector<Value *> AttentionOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> AttentionOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1, -1, -1, -1}; // Batch, heads, seq_len, dim
}

std::string AttentionOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 4 && Dims[0] != -1 && Dims[1] != -1 && Dims[2] != -1 && Dims[3] != -1) {
    return "Attention(" + std::to_string(Dims[0]) + "x" +
           std::to_string(Dims[1]) + "x" + std::to_string(Dims[2]) + "x" +
           std::to_string(Dims[3]) + ")";
  }
  return "Attention";
}

// LayerNormOp implementation
bool LayerNormOp::canFuseWith(const TensorOperation &Other) const {
  // Layer normalization can be fused with element-wise operations and activations
  return Other.getKind() == TensorOpKind::ElementWiseAdd ||
         Other.getKind() == TensorOpKind::ElementWiseMul ||
         Other.getKind() == TensorOpKind::Activation;
}

std::vector<Value *> LayerNormOp::getOperands() const {
  // For a layer normalization, we expect 1 operand
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (!Operands.empty()) {
    return {Operands[0]};
  }
  return Operands;
}

std::vector<Value *> LayerNormOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> LayerNormOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1, -1}; // Batch, hidden_size
}

std::string LayerNormOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 2 && Dims[0] != -1 && Dims[1] != -1) {
    return "LayerNorm(" + std::to_string(Dims[0]) + "x" +
           std::to_string(Dims[1]) + ")";
  }
  return "LayerNorm";
}

// SoftmaxOp implementation
bool SoftmaxOp::canFuseWith(const TensorOperation &Other) const {
  // Softmax can be fused with element-wise operations
  return Other.getKind() == TensorOpKind::ElementWiseAdd ||
         Other.getKind() == TensorOpKind::ElementWiseMul;
}

std::vector<Value *> SoftmaxOp::getOperands() const {
  // For a softmax, we expect 1 operand
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (!Operands.empty()) {
    return {Operands[0]};
  }
  return Operands;
}

std::vector<Value *> SoftmaxOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> SoftmaxOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1}; // Size
}

std::string SoftmaxOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 1 && Dims[0] != -1) {
    return "Softmax(" + std::to_string(Dims[0]) + ")";
  }
  return "Softmax";
}

// TensorContractionOp implementation
bool TensorContractionOp::canFuseWith(const TensorOperation &Other) const {
  // Tensor contraction can be fused with element-wise operations
  return Other.getKind() == TensorOpKind::ElementWiseAdd ||
         Other.getKind() == TensorOpKind::ElementWiseMul;
}

std::vector<Value *> TensorContractionOp::getOperands() const {
  // For a tensor contraction, we expect 2 operands
  std::vector<Value *> Operands = TensorOperation::getOperands();
  if (Operands.size() >= 2) {
    return {Operands[0], Operands[1]};
  }
  return Operands;
}

std::vector<Value *> TensorContractionOp::getResults() const {
  return TensorOperation::getResults();
}

std::vector<int64_t> TensorContractionOp::getDimensions() const {
  // Try to infer dimensions from the instruction
  // This is a placeholder implementation
  return {-1, -1, -1}; // Input dims, contracted dims, output dims
}

std::string TensorContractionOp::toString() const {
  auto Dims = getDimensions();
  if (Dims.size() == 3 && Dims[0] != -1 && Dims[1] != -1 && Dims[2] != -1) {
    return "TensorContraction(" + std::to_string(Dims[0]) + "," +
           std::to_string(Dims[1]) + "->" + std::to_string(Dims[2]) + ")";
  }
  return "TensorContraction";
}

// Register pattern matchers for the new tensor operations
namespace {

// Pattern matcher for element-wise add operations
bool isElementWiseAdd(Instruction *Inst) {
  if (auto *BinOp = dyn_cast<BinaryOperator>(Inst)) {
    if (BinOp->getOpcode() == Instruction::FAdd || BinOp->getOpcode() == Instruction::Add) {
      // Check if the operands have the same type and dimensions
      Type *LHSType = BinOp->getOperand(0)->getType();
      Type *RHSType = BinOp->getOperand(1)->getType();
      
      // For now, just check if the types are the same
      return LHSType == RHSType;
    }
  }
  
  // Check for direct calls to element-wise add functions
  if (auto *Call = dyn_cast<CallInst>(Inst)) {
    if (functionNameContains(Call, "add") ||
        functionNameContains(Call, "element_wise_add")) {
      return true;
    }
  }
  
  return false;
}

// Pattern matcher for element-wise mul operations
bool isElementWiseMul(Instruction *Inst) {
  if (auto *BinOp = dyn_cast<BinaryOperator>(Inst)) {
    if (BinOp->getOpcode() == Instruction::FMul || BinOp->getOpcode() == Instruction::Mul) {
      // Check if the operands have the same type and dimensions
      Type *LHSType = BinOp->getOperand(0)->getType();
      Type *RHSType = BinOp->getOperand(1)->getType();
      
      // For now, just check if the types are the same
      return LHSType == RHSType;
    }
  }
  
  // Check for direct calls to element-wise mul functions
  if (auto *Call = dyn_cast<CallInst>(Inst)) {
    if (functionNameContains(Call, "mul") ||
        functionNameContains(Call, "element_wise_mul")) {
      return true;
    }
  }
  
  return false;
}

// Register the new tensor operations
static RegisterTensorOperation<ElementWiseAddOp> RegisterElementWiseAdd("ElementWiseAdd", isElementWiseAdd);
static RegisterTensorOperation<ElementWiseMulOp> RegisterElementWiseMul("ElementWiseMul", isElementWiseMul);

} // anonymous namespace

} // namespace tensor
} // namespace llvm
