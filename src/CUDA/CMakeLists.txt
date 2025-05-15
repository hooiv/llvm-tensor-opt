if(CUDA_ENABLED)
  cuda_add_library(LLVMTensorCUDA
    TensorKernels.cu
    CUDAOffloader.cpp
    SHARED
  )

  target_link_libraries(LLVMTensorCUDA
    PRIVATE
    ${CUDA_LIBRARIES}
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
