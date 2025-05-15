#include "Transforms/TensorFusion.h"
#include "Transforms/TensorVectorization.h"
#include "Transforms/TensorParallelization.h"
#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorAccessPatternAnalysis.h"
#include "CUDA/CUDAOffloader.h"

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

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace llvm;
using namespace std::chrono;

// Command line options
static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input bitcode file>"), cl::Required);
static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));
static cl::opt<std::string> BenchmarkOutputFilename("benchmark-output", cl::desc("Benchmark output filename"), cl::value_desc("filename"));

static cl::opt<bool> EnableFusion("fusion", cl::desc("Enable tensor fusion optimization"));
static cl::opt<bool> EnableVectorization("vectorization", cl::desc("Enable tensor vectorization optimization"));
static cl::opt<bool> EnableParallelization("parallelization", cl::desc("Enable tensor parallelization optimization"));
static cl::opt<bool> EnableCUDAOffload("cuda-offload", cl::desc("Enable CUDA offloading"));

static cl::opt<int> NumRuns("num-runs", cl::desc("Number of benchmark runs"), cl::init(10));

// Benchmark results
struct BenchmarkResult {
  std::string PassName;
  double TimeMs;
  
  BenchmarkResult(const std::string& PassName, double TimeMs)
    : PassName(PassName), TimeMs(TimeMs) {}
};

// Run a pass and measure its execution time
template <typename PassT>
BenchmarkResult runPass(PassT& Pass, Function& F, FunctionAnalysisManager& FAM, const std::string& PassName) {
  // Warm up
  Pass.run(F, FAM);
  
  // Benchmark
  std::vector<double> Times;
  for (int i = 0; i < NumRuns; ++i) {
    auto Start = high_resolution_clock::now();
    Pass.run(F, FAM);
    auto End = high_resolution_clock::now();
    
    double TimeMs = duration_cast<microseconds>(End - Start).count() / 1000.0;
    Times.push_back(TimeMs);
  }
  
  // Calculate average time
  double AvgTimeMs = std::accumulate(Times.begin(), Times.end(), 0.0) / Times.size();
  
  return BenchmarkResult(PassName, AvgTimeMs);
}

int main(int argc, char **argv) {
  // Parse command line options
  cl::ParseCommandLineOptions(argc, argv, "LLVM Tensor Benchmark Tool\n");
  
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
  
  // Create our custom optimization passes
  tensor::TensorFusionPass FusionPass;
  tensor::TensorVectorizationPass VectorizationPass;
  tensor::TensorParallelizationPass ParallelizationPass;
  
  // Benchmark results
  std::vector<BenchmarkResult> Results;
  
  // Benchmark each function in the module
  for (auto& F : *M) {
    if (F.isDeclaration())
      continue;
    
    outs() << "Benchmarking function: " << F.getName() << "\n";
    
    // Benchmark analysis passes
    Results.push_back(runPass(tensor::TensorDataFlowAnalysis(), F, FAM, "TensorDataFlowAnalysis"));
    Results.push_back(runPass(tensor::TensorAccessPatternAnalysis(), F, FAM, "TensorAccessPatternAnalysis"));
    
    // Benchmark optimization passes
    if (EnableFusion) {
      Results.push_back(runPass(FusionPass, F, FAM, "TensorFusion"));
    }
    
    if (EnableVectorization) {
      Results.push_back(runPass(VectorizationPass, F, FAM, "TensorVectorization"));
    }
    
    if (EnableParallelization) {
      Results.push_back(runPass(ParallelizationPass, F, FAM, "TensorParallelization"));
    }
  }
  
  // Benchmark CUDA offloading
  if (EnableCUDAOffload) {
    tensor::cuda::CUDAOffloader Offloader;
    
    // Warm up
    Offloader.offloadModule(*M);
    
    // Benchmark
    std::vector<double> Times;
    for (int i = 0; i < NumRuns; ++i) {
      auto Start = high_resolution_clock::now();
      Offloader.offloadModule(*M);
      auto End = high_resolution_clock::now();
      
      double TimeMs = duration_cast<microseconds>(End - Start).count() / 1000.0;
      Times.push_back(TimeMs);
    }
    
    // Calculate average time
    double AvgTimeMs = std::accumulate(Times.begin(), Times.end(), 0.0) / Times.size();
    
    Results.push_back(BenchmarkResult("CUDAOffload", AvgTimeMs));
  }
  
  // Print benchmark results
  outs() << "\nBenchmark Results:\n";
  outs() << "=================\n";
  for (const auto& Result : Results) {
    outs() << Result.PassName << ": " << Result.TimeMs << " ms\n";
  }
  
  // Write benchmark results to file if requested
  if (!BenchmarkOutputFilename.empty()) {
    std::error_code EC;
    raw_fd_ostream OS(BenchmarkOutputFilename, EC, sys::fs::OF_None);
    if (EC) {
      errs() << "Error opening benchmark output file: " << EC.message() << "\n";
      return 1;
    }
    
    OS << "Pass,TimeMs\n";
    for (const auto& Result : Results) {
      OS << Result.PassName << "," << Result.TimeMs << "\n";
    }
  }
  
  // Apply all enabled passes to the module
  FunctionPassManager FPM;
  
  if (EnableFusion) {
    FPM.addPass(tensor::TensorFusionPass());
  }
  
  if (EnableVectorization) {
    FPM.addPass(tensor::TensorVectorizationPass());
  }
  
  if (EnableParallelization) {
    FPM.addPass(tensor::TensorParallelizationPass());
  }
  
  // Create a module pass manager
  ModulePassManager MPM;
  
  // Add the function pass manager to the module pass manager
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  
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
