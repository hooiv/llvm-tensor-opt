enable_testing()

# Find Google Test
find_package(GTest QUIET)
if(GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS})
else()
  message(STATUS "GTest not found, downloading from source")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

# Unit tests
add_executable(unit_tests
  TransformsTest.cpp
  AnalysisTest.cpp
)

target_link_libraries(unit_tests
  PRIVATE
  LLVMTensorTransforms
  LLVMTensorAnalysis
  LLVMTensorCUDA
  gtest
  gtest_main
  LLVMCore
  LLVMSupport
)

add_test(NAME UnitTests COMMAND unit_tests)

# Integration tests
add_executable(integration_tests
  IntegrationTest.cpp
)

target_link_libraries(integration_tests
  PRIVATE
  LLVMTensorTransforms
  LLVMTensorAnalysis
  LLVMTensorCUDA
  gtest
  gtest_main
  LLVMCore
  LLVMSupport
)

add_test(NAME IntegrationTests COMMAND integration_tests)
