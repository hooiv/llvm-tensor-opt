#ifndef LLVM_TENSOR_OPT_MLIR_TENSOR_OPS_H
#define LLVM_TENSOR_OPT_MLIR_TENSOR_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir {
namespace tensor_opt {

#define GET_OP_CLASSES
#include "MLIR/TensorOps.h.inc"

} // namespace tensor_opt
} // namespace mlir

#endif // LLVM_TENSOR_OPT_MLIR_TENSOR_OPS_H
