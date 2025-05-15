#ifndef LLVM_TENSOR_OPT_MLIR_TENSOR_DIALECT_H
#define LLVM_TENSOR_OPT_MLIR_TENSOR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace tensor_opt {

class TensorDialect : public Dialect {
public:
  explicit TensorDialect(MLIRContext *context);
  
  static StringRef getDialectNamespace() { return "tensor_opt"; }
  
  void initialize();
};

} // namespace tensor_opt
} // namespace mlir

#endif // LLVM_TENSOR_OPT_MLIR_TENSOR_DIALECT_H
