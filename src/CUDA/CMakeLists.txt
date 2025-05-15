# Use modern CUDA approach with FindCUDA deprecated
if(CUDA_ENABLED)
  # Use modern CMake approach for CUDA
  enable_language(CUDA)

  add_library(LLVMTensorCUDA SHARED
    CUDAOffloader.cpp
  )

  # Add CUDA file separately to handle compilation options
  set_source_files_properties(TensorKernels.cu PROPERTIES LANGUAGE CUDA)
  target_sources(LLVMTensorCUDA PRIVATE TensorKernels.cu)

  # Set CUDA specific options
  set_target_properties(LLVMTensorCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  target_link_libraries(LLVMTensorCUDA
    PRIVATE
    CUDA::cudart
    LLVMCore
    LLVMSupport
  )
else()
  add_library(LLVMTensorCUDA
    CUDAOffloader.cpp
  )

  target_link_libraries(LLVMTensorCUDA
    PRIVATE
    LLVMCore
    LLVMSupport
  )
endif()
