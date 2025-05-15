#ifndef LLVM_TENSOR_OPT_MLIR_MLIR_TENSOR_TO_LLVM_H
#define LLVM_TENSOR_OPT_MLIR_MLIR_TENSOR_TO_LLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace tensor_opt {

// Create a pass to convert tensor dialect operations to LLVM dialect
std::unique_ptr<Pass> createConvertTensorToLLVMPass();

// Add patterns to convert tensor dialect operations to LLVM dialect
void populateTensorToLLVMConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context);

} // namespace tensor_opt
} // namespace mlir

#endif // LLVM_TENSOR_OPT_MLIR_MLIR_TENSOR_TO_LLVM_H
