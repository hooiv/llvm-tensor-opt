#include "tensor_jit.h"

#include "Transforms/TensorFusion.h"
#include "Transforms/TensorVectorization.h"
#include "Transforms/TensorParallelization.h"
#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorAccessPatternAnalysis.h"
#include "CUDA/CUDAOffloader.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/Passes/PassBuilder.h"

#include <sstream>

// External declarations of optimization flags
extern bool EnableFusion;
extern bool EnableVectorization;
extern bool EnableParallelization;
extern bool EnableCUDAOffload;

TensorJIT::TensorJIT()
  : Context(std::make_unique<llvm::LLVMContext>()) {
  // Initialize LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  
  // Create the execution engine
  Engine = std::unique_ptr<llvm::ExecutionEngine>(
    llvm::EngineBuilder(std::unique_ptr<llvm::Module>(new llvm::Module("jit", *Context)))
      .setEngineKind(llvm::EngineKind::JIT)
      .create()
  );
}

TensorJIT::~TensorJIT() = default;

bool TensorJIT::addModule(const std::string &ir) {
  // Parse the IR string
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> M = llvm::parseIR(
    llvm::MemoryBufferRef(ir, "module"),
    Err,
    *Context
  );
  
  if (!M) {
    return false;
  }
  
  // Add the module to the JIT
  std::string ModuleName = M->getName().str();
  Modules[ModuleName] = std::move(M);
  
  return true;
}

bool TensorJIT::optimizeModule(const std::string &moduleName) {
  // Find the module
  auto It = Modules.find(moduleName);
  if (It == Modules.end()) {
    return false;
  }
  
  // Apply optimization passes
  return applyOptimizationPasses(*It->second);
}

torch::Tensor TensorJIT::run(const std::string &funcName, const std::vector<torch::Tensor> &inputs) {
  // Find the function
  llvm::Function *Func = nullptr;
  for (const auto &Entry : Modules) {
    if (llvm::Function *F = Entry.second->getFunction(funcName)) {
      Func = F;
      break;
    }
  }
  
  if (!Func) {
    return torch::Tensor();
  }
  
  // Get the function pointer
  using FuncType = void (*)(float*, float*, int, int, int);
  FuncType FuncPtr = reinterpret_cast<FuncType>(Engine->getPointerToFunction(Func));
  
  // Prepare inputs and output
  torch::Tensor output = torch::zeros_like(inputs[0]);
  
  // Call the function
  FuncPtr(
    inputs[0].data_ptr<float>(),
    output.data_ptr<float>(),
    inputs[0].size(0),
    inputs[0].size(1),
    inputs[0].size(2)
  );
  
  return output;
}

bool TensorJIT::applyOptimizationPasses(llvm::Module &M) {
  // Create a new pass manager
  llvm::PassBuilder PB;
  
  // Create analysis managers
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  
  // Register the analysis passes
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  
  // Register our custom analysis passes
  FAM.registerPass([&] { return llvm::tensor::TensorDataFlowAnalysis(); });
  FAM.registerPass([&] { return llvm::tensor::TensorAccessPatternAnalysis(); });
  
  // Create a function pass manager
  llvm::FunctionPassManager FPM;
  
  // Add our custom optimization passes
  if (EnableFusion) {
    FPM.addPass(llvm::tensor::TensorFusionPass());
  }
  
  if (EnableVectorization) {
    FPM.addPass(llvm::tensor::TensorVectorizationPass());
  }
  
  if (EnableParallelization) {
    FPM.addPass(llvm::tensor::TensorParallelizationPass());
  }
  
  // Create a module pass manager
  llvm::ModulePassManager MPM;
  
  // Add the function pass manager to the module pass manager
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
  
  // Run the passes
  MPM.run(M, MAM);
  
  // Offload to CUDA if requested
  if (EnableCUDAOffload) {
    llvm::tensor::cuda::CUDAOffloader Offloader;
    Offloader.offloadModule(M);
  }
  
  return true;
}

template <typename FuncType>
FuncType TensorJIT::getFunction(const std::string &funcName) {
  return reinterpret_cast<FuncType>(Engine->getPointerToNamedFunction(funcName));
}
