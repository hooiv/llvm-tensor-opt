add_executable(tensor_opt_tool
  TensorOptTool.cpp
)

target_link_libraries(tensor_opt_tool
  PRIVATE
  LLVMTensorTransforms
  LLVMTensorAnalysis
  LLVMTensorCUDA
  LLVMTensorAutoTuning
  LLVMCore
  LLVMSupport
  LLVMIRReader
  LLVMPasses
  LLVMExecutionEngine
  # Add LLVM target libraries for native target initialization
  LLVMX86Info
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Desc
  LLVMTarget
  LLVMAnalysis
  LLVMBitWriter
  LLVMBitReader
  LLVMSelectionDAG
  LLVMAsmPrinter
  LLVMMCParser
  LLVMMC
)

# Add MLIR libraries if MLIR is enabled
if(MLIR_ENABLED)
  target_link_libraries(tensor_opt_tool
    PRIVATE
    LLVMTensorMLIR
    MLIRIR
    MLIRLLVMDialect
  )
endif()

# Benchmark executable
add_executable(tensor_benchmark
  TensorBenchmark.cpp
)

target_link_libraries(tensor_benchmark
  PRIVATE
  LLVMTensorTransforms
  LLVMTensorAnalysis
  LLVMTensorCUDA
  LLVMTensorAutoTuning
  LLVMCore
  LLVMSupport
  LLVMIRReader
  LLVMExecutionEngine
  # Add LLVM target libraries for native target initialization
  LLVMX86Info
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Desc
  LLVMTarget
  LLVMAnalysis
  LLVMBitWriter
  LLVMBitReader
  LLVMSelectionDAG
  LLVMAsmPrinter
  LLVMMCParser
  LLVMMC
  # Add LLVM pass manager libraries
  LLVMPasses
  LLVMTransformUtils
  LLVMScalarOpts
  LLVMipo
  LLVMAggressiveInstCombine
  LLVMInstCombine
  LLVMInstrumentation
  LLVMVectorize
  LLVMObjCARCOpts
  LLVMCoroutines
  LLVMCFGuard
  LLVMLinker
)

# Add MLIR libraries if MLIR is enabled
if(MLIR_ENABLED)
  target_link_libraries(tensor_benchmark
    PRIVATE
    LLVMTensorMLIR
    MLIRIR
    MLIRLLVMDialect
  )
endif()

# Example tensor operations
add_executable(matrix_multiply
  MatrixMultiply.cpp
)

if(CUDA_ENABLED)
  target_link_libraries(matrix_multiply
    PRIVATE
    LLVMTensorCUDA
    ${CUDA_LIBRARIES}
  )
endif()
