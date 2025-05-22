#include "Transforms/TensorFusion.h"
#include "Transforms/TensorVectorization.h"
#include "Transforms/TensorParallelization.h"
#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorAccessPatternAnalysis.h"
#include "Analysis/TensorOperationRegistry.h"
#include "Analysis/TensorOperations.h"
#include "CUDA/CUDAOffloader.h"
#include "AutoTuning/AutoTuner.h"
#include "AutoTuning/CostModel.h"
#include "AutoTuning/Profiler.h"

// Include MLIR headers only if MLIR is enabled
#ifdef MLIR_ENABLED
#include "MLIR/MLIRTensorToLLVM.h"
#endif

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/LegacyPassManager.h"

using namespace llvm;

// Command line options
static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input bitcode file>"), cl::Required);
static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));

static cl::opt<bool> EnableFusion("fusion", cl::desc("Enable tensor fusion optimization"));
static cl::opt<bool> EnableVectorization("vectorization", cl::desc("Enable tensor vectorization optimization"));
static cl::opt<bool> EnableParallelization("parallelization", cl::desc("Enable tensor parallelization optimization"));
static cl::opt<bool> EnableCUDAOffload("cuda-offload", cl::desc("Enable CUDA offloading"));
static cl::opt<bool> EnableAutoTuning("auto-tune", cl::desc("Enable auto-tuning"));
static cl::opt<bool> EnableMLCostModel("ml-cost-model", cl::desc("Enable machine learning-based cost model"));
static cl::opt<bool> EnableProfiling("profile", cl::desc("Enable profiling-based optimization selection"));
static cl::opt<bool> EnableMLIR("mlir", cl::desc("Enable MLIR-based optimizations"));

int main(int argc, char **argv) {
  // Parse command line options
  cl::ParseCommandLineOptions(argc, argv, "LLVM Tensor Optimization Tool\n");

  // Create LLVM context
  LLVMContext Context;
  SMDiagnostic Err;

  // Load the input module
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // Create a new pass manager
  PassBuilder PB;

  // Create analysis managers
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register the analysis passes
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Register our custom analysis passes
  FAM.registerPass([&] { return tensor::TensorDataFlowAnalysis(); });
  FAM.registerPass([&] { return tensor::TensorAccessPatternAnalysis(); });

  // Create a module pass manager
  ModulePassManager MPM;

  if (EnableAutoTuning) {
    // Use auto-tuning to select the best optimization strategy
    outs() << "Using auto-tuning to select the best optimization strategy\n";

    // Create an auto-tuner
    tensor::AutoTuner Tuner(EnableMLCostModel);

    // Apply auto-tuning to each function in the module
    for (auto &F : *M) {
      if (F.isDeclaration())
        continue;

      outs() << "Auto-tuning function: " << F.getName() << "\n";

      // Tune the function
      Tuner.tune(F, FAM);
    }
  } else if (EnableProfiling) {
    // Use profiling to select the best optimization strategy
    outs() << "Using profiling to select the best optimization strategy\n";

    // Create a profiler
    tensor::Profiler Profiler;

    // Apply profiling to each function in the module
    for (auto &F : *M) {
      if (F.isDeclaration())
        continue;

      outs() << "Profiling function: " << F.getName() << "\n";

      // Define the optimization strategies to profile
      std::vector<tensor::OptimizationStrategy> Strategies = {
        tensor::OptimizationStrategy::None,
        tensor::OptimizationStrategy::Fusion,
        tensor::OptimizationStrategy::Vectorization,
        tensor::OptimizationStrategy::Parallelization,
        tensor::OptimizationStrategy::All
      };

      // Profile the function with different optimization strategies
      auto Results = Profiler.profileFunction(F, Strategies);

      // Get the best optimization strategy
      auto BestStrategy = Profiler.getBestStrategy(Results);

      outs() << "Best strategy for function " << F.getName()
             << " is " << static_cast<int>(BestStrategy) << "\n";

      // Create an auto-tuner with the selected strategy
      tensor::AutoTuner Tuner;

      // Apply the best strategy
      Tuner.applyStrategy(F, FAM, BestStrategy);
    }
  } else {
    // Use the specified optimization passes
    FunctionPassManager FPM;

    // Add our custom optimization passes
    if (EnableFusion) {
      FPM.addPass(tensor::TensorFusionPass());
    }

    if (EnableVectorization) {
      FPM.addPass(tensor::TensorVectorizationPass());
    }

    if (EnableParallelization) {
      FPM.addPass(tensor::TensorParallelizationPass());
    }

    // Add the function pass manager to the module pass manager
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  // Add MLIR-based optimizations if requested and available
  if (EnableMLIR) {
#ifdef MLIR_ENABLED
    outs() << "Adding MLIR-based optimizations\n";

    // Add the MLIR tensor to LLVM conversion pass
    MPM.addPass(mlir::tensor_opt::createConvertTensorToLLVMPass());
#else
    outs() << "MLIR support is not available. Skipping MLIR-based optimizations.\n";
#endif
  }

  // Run the passes
  MPM.run(*M, MAM);

  // Offload to CUDA if requested
  if (EnableCUDAOffload) {
    tensor::cuda::CUDAOffloader Offloader;
    Offloader.offloadModule(*M);
  }

  // Write the transformed module to the output file
  if (!OutputFilename.empty()) {
    std::error_code EC;
    raw_fd_ostream OS(OutputFilename, EC, sys::fs::OF_None);
    if (EC) {
      errs() << "Error opening output file: " << EC.message() << "\n";
      return 1;
    }
    OS << *M;
  } else {
    outs() << *M;
  }

  return 0;
}
