#ifndef LLVM_TENSOR_OPT_ANALYSIS_TENSOR_OPERATION_REGISTRY_H
#define LLVM_TENSOR_OPT_ANALYSIS_TENSOR_OPERATION_REGISTRY_H

#include "llvm/IR/Instructions.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
namespace tensor {

/// Enum representing different tensor operation types
enum class TensorOpKind {
  Unknown,
  MatrixMultiply,
  Convolution,
  ElementWiseAdd,
  ElementWiseMul,
  Reduction,
  Transpose,
  Reshape,
  Concat,
  Split,
  Attention,
  LayerNorm,
  Softmax,
  Pooling,
  BatchNorm,
  Activation,
  TensorContraction
};

/// Class representing a tensor operation
class TensorOperation {
public:
  TensorOperation(TensorOpKind Kind, Instruction *Inst)
      : Kind(Kind), Inst(Inst) {}
  
  virtual ~TensorOperation() = default;
  
  /// Get the kind of tensor operation
  TensorOpKind getKind() const { return Kind; }
  
  /// Get the instruction that represents this tensor operation
  Instruction *getInstruction() const { return Inst; }
  
  /// Get the name of this tensor operation
  virtual StringRef getName() const;
  
  /// Check if this tensor operation can be fused with another
  virtual bool canFuseWith(const TensorOperation &Other) const;
  
  /// Check if this tensor operation can be vectorized
  virtual bool canVectorize() const;
  
  /// Check if this tensor operation can be parallelized
  virtual bool canParallelize() const;
  
  /// Get the operands of this tensor operation
  virtual std::vector<Value *> getOperands() const;
  
  /// Get the results of this tensor operation
  virtual std::vector<Value *> getResults() const;
  
  /// Get the dimensions of this tensor operation
  virtual std::vector<int64_t> getDimensions() const;
  
  /// Create a string representation of this tensor operation
  virtual std::string toString() const;
  
  /// Check if this is a specific kind of tensor operation
  template <typename T>
  bool is() const {
    return Kind == T::getClassKind();
  }
  
  /// Cast this tensor operation to a specific type
  template <typename T>
  const T *as() const {
    if (is<T>())
      return static_cast<const T *>(this);
    return nullptr;
  }
  
  /// Cast this tensor operation to a specific type
  template <typename T>
  T *as() {
    if (is<T>())
      return static_cast<T *>(this);
    return nullptr;
  }

protected:
  TensorOpKind Kind;
  Instruction *Inst;
};

/// Class for matrix multiplication operations
class MatrixMultiplyOp : public TensorOperation {
public:
  MatrixMultiplyOp(Instruction *Inst) : TensorOperation(TensorOpKind::MatrixMultiply, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::MatrixMultiply; }
  
  StringRef getName() const override { return "MatrixMultiply"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Class for convolution operations
class ConvolutionOp : public TensorOperation {
public:
  ConvolutionOp(Instruction *Inst) : TensorOperation(TensorOpKind::Convolution, Inst) {}
  
  static TensorOpKind getClassKind() { return TensorOpKind::Convolution; }
  
  StringRef getName() const override { return "Convolution"; }
  
  bool canFuseWith(const TensorOperation &Other) const override;
  bool canVectorize() const override { return true; }
  bool canParallelize() const override { return true; }
  
  std::vector<Value *> getOperands() const override;
  std::vector<Value *> getResults() const override;
  std::vector<int64_t> getDimensions() const override;
  std::string toString() const override;
};

/// Factory function to create a tensor operation from an instruction
std::unique_ptr<TensorOperation> createTensorOperation(Instruction *Inst);

/// Class for registering tensor operation patterns
class TensorOperationRegistry {
public:
  using PatternMatcherTy = std::function<bool(Instruction *)>;
  using TensorOpCreatorTy = std::function<std::unique_ptr<TensorOperation>(Instruction *)>;
  
  struct PatternEntry {
    PatternMatcherTy Matcher;
    TensorOpCreatorTy Creator;
    StringRef Name;
    
    PatternEntry(PatternMatcherTy Matcher, TensorOpCreatorTy Creator, StringRef Name)
        : Matcher(std::move(Matcher)), Creator(std::move(Creator)), Name(Name) {}
  };
  
  /// Get the singleton instance of the registry
  static TensorOperationRegistry &getInstance();
  
  /// Register a pattern for a tensor operation
  void registerPattern(PatternMatcherTy Matcher, TensorOpCreatorTy Creator, StringRef Name);
  
  /// Match an instruction to a tensor operation
  std::unique_ptr<TensorOperation> matchAndCreate(Instruction *Inst) const;
  
  /// Get all registered patterns
  const std::vector<PatternEntry> &getPatterns() const { return Patterns; }

private:
  TensorOperationRegistry() = default;
  
  std::vector<PatternEntry> Patterns;
};

/// Helper class for registering tensor operation patterns
template <typename OpTy>
class RegisterTensorOperation {
public:
  RegisterTensorOperation(StringRef Name, TensorOperationRegistry::PatternMatcherTy Matcher) {
    TensorOperationRegistry::getInstance().registerPattern(
        std::move(Matcher),
        [](Instruction *Inst) { return std::make_unique<OpTy>(Inst); },
        Name);
  }
};

} // namespace tensor
} // namespace llvm

#endif // LLVM_TENSOR_OPT_ANALYSIS_TENSOR_OPERATION_REGISTRY_H
