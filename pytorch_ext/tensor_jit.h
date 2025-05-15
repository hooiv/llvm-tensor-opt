#ifndef LLVM_TENSOR_OPT_PYTORCH_TENSOR_JIT_H
#define LLVM_TENSOR_OPT_PYTORCH_TENSOR_JIT_H

#include <torch/torch.h>
#include <string>
#include <memory>
#include <vector>

// Forward declarations
namespace llvm {
class Module;
class LLVMContext;
class ExecutionEngine;
}

// Class for JIT compiling and executing tensor operations
class TensorJIT {
public:
  TensorJIT();
  ~TensorJIT();
  
  // Add a module to the JIT
  bool addModule(const std::string &ir);
  
  // Optimize a module in the JIT
  bool optimizeModule(const std::string &moduleName);
  
  // Run a function in the JIT
  torch::Tensor run(const std::string &funcName, const std::vector<torch::Tensor> &inputs);
  
private:
  // LLVM context
  std::unique_ptr<llvm::LLVMContext> Context;
  
  // LLVM execution engine
  std::unique_ptr<llvm::ExecutionEngine> Engine;
  
  // Map of module names to modules
  std::unordered_map<std::string, std::unique_ptr<llvm::Module>> Modules;
  
  // Apply tensor optimization passes to a module
  bool applyOptimizationPasses(llvm::Module &M);
  
  // Create a function pointer for a JIT-compiled function
  template <typename FuncType>
  FuncType getFunction(const std::string &funcName);
};

#endif // LLVM_TENSOR_OPT_PYTORCH_TENSOR_JIT_H
