add_library(LLVMTensorAutoTuning
  CostModel.cpp
  AutoTuner.cpp
  Profiler.cpp
)

target_link_libraries(LLVMTensorAutoTuning
  PRIVATE
  LLVMTensorTransforms
  LLVMTensorAnalysis
  LLVMCore
  LLVMSupport
  LLVMExecutionEngine
)
