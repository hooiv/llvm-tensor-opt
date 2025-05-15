add_library(LLVMTensorAnalysis
  TensorDataFlowAnalysis.cpp
  TensorAccessPatternAnalysis.cpp
  TensorOperationRegistry.cpp
  TensorOperations.cpp
)

target_link_libraries(LLVMTensorAnalysis
  PRIVATE
  LLVMAnalysis
  LLVMCore
  LLVMSupport
)
