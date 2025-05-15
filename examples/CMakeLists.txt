add_executable(tensor_opt_tool
  TensorOptTool.cpp
)

target_link_libraries(tensor_opt_tool
  PRIVATE
  LLVMTensorTransforms
  LLVMTensorAnalysis
  LLVMTensorCUDA
  LLVMTensorMLIR
  LLVMTensorAutoTuning
  LLVMCore
  LLVMSupport
  LLVMIRReader
  LLVMPasses
  LLVMExecutionEngine
  MLIRIR
  MLIRLLVMDialect
)

# Benchmark executable
add_executable(tensor_benchmark
  TensorBenchmark.cpp
)

target_link_libraries(tensor_benchmark
  PRIVATE
  LLVMTensorTransforms
  LLVMTensorAnalysis
  LLVMTensorCUDA
  LLVMTensorMLIR
  LLVMTensorAutoTuning
  LLVMCore
  LLVMSupport
  LLVMIRReader
  LLVMExecutionEngine
  MLIRIR
  MLIRLLVMDialect
)

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
