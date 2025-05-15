#include "MLIR/TensorOps.h"
#include "MLIR/TensorDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace tensor_opt {

#define GET_OP_CLASSES
#include "MLIR/TensorOps.cpp.inc"

} // namespace tensor_opt
} // namespace mlir
