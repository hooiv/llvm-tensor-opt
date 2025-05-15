add_library(LLVMTensorMLIR
  TensorDialect.cpp
  TensorOps.cpp
  MLIRTensorToLLVM.cpp
)

target_link_libraries(LLVMTensorMLIR
  PRIVATE
  LLVMCore
  LLVMSupport
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)

# Generate dialect operation definitions
set(LLVM_TARGET_DEFINITIONS TensorOps.td)
mlir_tablegen(TensorOps.h.inc -gen-op-decls)
mlir_tablegen(TensorOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTensorOpsIncGen)
