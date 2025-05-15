if(MLIR_ENABLED)
  # Generate dialect operation definitions
  set(LLVM_TARGET_DEFINITIONS TensorOps.td)
  mlir_tablegen(TensorOps.h.inc -gen-op-decls)
  mlir_tablegen(TensorOps.cpp.inc -gen-op-defs)
  add_public_tablegen_target(MLIRTensorOpsIncGen)

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
else()
  # Create a stub library when MLIR is not available
  add_library(LLVMTensorMLIR INTERFACE)
  message(STATUS "Building without MLIR support")
endif()
