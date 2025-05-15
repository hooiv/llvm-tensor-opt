#include "MLIR/TensorDialect.h"
#include "MLIR/TensorOps.h"

using namespace mlir;
using namespace mlir::tensor_opt;

TensorDialect::TensorDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TensorDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "MLIR/TensorOps.cpp.inc"
      >();
}

void TensorDialect::initialize() {
  // No additional initialization required
}
