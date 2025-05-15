#ifndef LLVM_TENSOR_OPT_PYTORCH_TENSOR_COMPILER_H
#define LLVM_TENSOR_OPT_PYTORCH_TENSOR_COMPILER_H

#include <torch/torch.h>
#include <string>
#include <memory>

// Forward declarations
namespace llvm {
class Module;
class LLVMContext;
}

// Flags for enabling different optimizations
void enableFusion(bool Enable);
void enableVectorization(bool Enable);
void enableParallelization(bool Enable);
void enableCUDAOffload(bool Enable);

// Class for compiling PyTorch tensor operations to LLVM IR
class TensorCompiler {
public:
  TensorCompiler();
  ~TensorCompiler();
  
  // Compile a PyTorch tensor operation to LLVM IR
  bool compile(const torch::Tensor &input, const std::string &opName);
  
  // Optimize the compiled LLVM IR
  bool optimize();
  
  // Get the optimized LLVM IR
  std::string getIR() const;
  
private:
  // LLVM context and module
  std::unique_ptr<llvm::LLVMContext> Context;
  std::unique_ptr<llvm::Module> Module;
  
  // Compiled function name
  std::string FunctionName;
  
  // Convert a PyTorch tensor to LLVM IR
  bool convertTensorToIR(const torch::Tensor &tensor, const std::string &opName);
  
  // Apply tensor optimization passes
  bool applyOptimizationPasses();
};

#endif // LLVM_TENSOR_OPT_PYTORCH_TENSOR_COMPILER_H
