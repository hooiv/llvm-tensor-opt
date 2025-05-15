cmake_minimum_required(VERSION 3.13.4)
project(LLVMTensorOpt)

# Find LLVM package
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find MLIR package
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_PREFIX}/lib/cmake/mlir")
find_package(MLIR QUIET CONFIG)
if(MLIR_FOUND)
  message(STATUS "Found MLIR ${MLIR_VERSION}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  # Add MLIR includes and definitions
  include_directories(${MLIR_INCLUDE_DIRS})
  add_definitions(${MLIR_DEFINITIONS})

  # Add MLIR tablegen support
  include(AddMLIR)

  set(MLIR_ENABLED TRUE)
  add_definitions(-DMLIR_ENABLED)
else()
  message(STATUS "MLIR not found, MLIR features will be disabled")
  set(MLIR_ENABLED FALSE)
endif()

# LLVM settings
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})

# Find CUDA using modern CMake approach
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")
    set(CUDA_ENABLED TRUE)
    add_definitions(-DCUDA_ENABLED)
  else()
    message(STATUS "CUDA Toolkit not found, GPU features will be disabled")
    set(CUDA_ENABLED FALSE)
  endif()
else()
  message(STATUS "CUDA compiler not found, GPU features will be disabled")
  set(CUDA_ENABLED FALSE)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)

# Installation settings
install(DIRECTORY include/ DESTINATION include/llvm-tensor-opt)
