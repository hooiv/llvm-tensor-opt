cmake_minimum_required(VERSION 3.13.4)
project(LLVMTensorOpt)

# Find LLVM package
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# LLVM settings
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})

# Find CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
  message(STATUS "Found CUDA ${CUDA_VERSION}")
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CUDA_ENABLED TRUE)
  add_definitions(-DCUDA_ENABLED)
else()
  message(STATUS "CUDA not found, GPU features will be disabled")
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
