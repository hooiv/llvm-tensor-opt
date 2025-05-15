add_library(LLVMTensorTransforms
  TensorFusion.cpp
  TensorVectorization.cpp
  TensorParallelization.cpp
)

target_link_libraries(LLVMTensorTransforms
  PRIVATE
  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMTransformUtils
)
