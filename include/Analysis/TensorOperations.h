#ifndef LLVM_TENSOR_OPT_ANALYSIS_TENSOR_OPERATIONS_H
#define LLVM_TENSOR_OPT_ANALYSIS_TENSOR_OPERATIONS_H

#include "Analysis/TensorOperationRegistry.h"

namespace llvm {
namespace tensor {

/// Class for element-wise addition operations
class ElementWiseAddOp : public TensorOperation {
public:
  ElementWiseAddOp(Instruction *Inst) : TensorOperation(TensorOpKind::ElementWiseAdd, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::ElementWiseAdd; }
  
  StringRef getName() const override { return "ElementWiseAdd"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for element-wise multiplication operations
class ElementWiseMulOp : public TensorOperation {
public:
  ElementWiseMulOp(Instruction *Inst) : TensorOperation(TensorOpKind::ElementWiseMul, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::ElementWiseMul; }
  
  StringRef getName() const override { return "ElementWiseMul"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for reduction operations
class ReductionOp : public TensorOperation {
public:
  ReductionOp(Instruction *Inst) : TensorOperation(TensorOpKind::Reduction, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::Reduction; }
  
  StringRef getName() const override { return "Reduction"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for transpose operations
class TransposeOp : public TensorOperation {
public:
  TransposeOp(Instruction *Inst) : TensorOperation(TensorOpKind::Transpose, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::Transpose; }
  
  StringRef getName() const override { return "Transpose"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for reshape operations
class ReshapeOp : public TensorOperation {
public:
  ReshapeOp(Instruction *Inst) : TensorOperation(TensorOpKind::Reshape, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::Reshape; }
  
  StringRef getName() const override { return "Reshape"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return false; }
  bool canParallelize() const override { return false; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for attention operations
class AttentionOp : public TensorOperation {
public:
  AttentionOp(Instruction *Inst) : TensorOperation(TensorOpKind::Attention, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::Attention; }
  
  StringRef getName() const override { return "Attention"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for layer normalization operations
class LayerNormOp : public TensorOperation {
public:
  LayerNormOp(Instruction *Inst) : TensorOperation(TensorOpKind::LayerNorm, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::LayerNorm; }
  
  StringRef getName() const override { return "LayerNorm"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for softmax operations
class SoftmaxOp : public TensorOperation {
public:
  SoftmaxOp(Instruction *Inst) : TensorOperation(TensorOpKind::Softmax, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::Softmax; }
  
  StringRef getName() const override { return "Softmax"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for tensor contraction operations
class TensorContractionOp : public TensorOperation {
public:
  TensorContractionOp(Instruction *Inst) : TensorOperation(TensorOpKind::TensorContraction, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::TensorContraction; }
  
  StringRef getName() const override { return "TensorContraction"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_ANALYSIS_TENSOR_OPERATIONS_H
