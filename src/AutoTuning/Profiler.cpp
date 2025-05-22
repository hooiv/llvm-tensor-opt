#include "AutoTuning/Profiler.h"
#include "AutoTuning/AutoTuner.h"
#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorAccessPatternAnalysis.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "profiler"

using namespace llvm;
using namespace llvm::tensor;

Profiler::Profiler() {
  // Initialize LLVM targets
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
}

Profiler::~Profiler() = default;

std::vector<std::pair<OptimizationStrategy, double>> Profiler::profileOperation(
    TensorOperation &Op, const std::vector<OptimizationStrategy> &Strategies) {
  std::vector<std::pair<OptimizationStrategy, double>> Results;

  // Get the instruction for this operation
  Instruction *Inst = Op.getInstruction();
  if (!Inst) {
    return Results;
  }

  // Get the function containing this instruction
  Function *F = Inst->getFunction();
  if (!F) {
    return Results;
  }

  // Profile the function with different optimization strategies
  return profileFunction(*F, Strategies);
}

std::vector<std::pair<OptimizationStrategy, double>> Profiler::profileFunction(
    Function &F, const std::vector<OptimizationStrategy> &Strategies) {
  std::vector<std::pair<OptimizationStrategy, double>> Results;

  // Create a copy of the function for each optimization strategy
  for (auto Strategy : Strategies) {
    // Apply the optimization strategy to the function
    Function *OptimizedF = applyStrategy(F, Strategy);
    if (!OptimizedF) {
      continue;
    }

    // Measure the execution time of the optimized function
    double ExecutionTime = measureExecutionTime(*OptimizedF);

    // Add the result
    Results.emplace_back(Strategy, ExecutionTime);

    LLVM_DEBUG(dbgs() << "Profiler: Strategy " << static_cast<int>(Strategy)
                      << " for function " << F.getName()
                      << " took " << ExecutionTime << " ms\n");
  }

  return Results;
}

OptimizationStrategy Profiler::getBestStrategy(
    const std::vector<std::pair<OptimizationStrategy, double>> &Results) const {
  if (Results.empty()) {
    return OptimizationStrategy::None;
  }

  // Find the strategy with the lowest execution time
  OptimizationStrategy BestStrategy = Results[0].first;
  double BestTime = Results[0].second;

  for (size_t i = 1; i < Results.size(); ++i) {
    if (Results[i].second < BestTime) {
      BestTime = Results[i].second;
      BestStrategy = Results[i].first;
    }
  }

  return BestStrategy;
}

Function *Profiler::applyStrategy(Function &F, OptimizationStrategy Strategy) {
  // Create a copy of the function
  Module *M = F.getParent();
  if (!M) {
    return nullptr;
  }

  // Create a new function with the same signature
  Function *NewF = Function::Create(
    F.getFunctionType(),
    F.getLinkage(),
    F.getName() + ".opt" + std::to_string(static_cast<int>(Strategy)),
    M
  );

  // Copy the function body by iterating through basic blocks
  ValueToValueMapTy VMap;
  for (const BasicBlock &BB : F) {
    // Create a new basic block in the new function
    BasicBlock *NewBB = BasicBlock::Create(F.getContext(), BB.getName(), NewF);
    VMap[&BB] = NewBB;

    // Clone instructions
    for (const Instruction &I : BB) {
      Instruction *NewInst = I.clone();
      if (NewInst->getType()->isVoidTy()) {
        NewBB->getInstList().push_back(NewInst);
      } else {
        NewBB->getInstList().push_back(NewInst);
        VMap[&I] = NewInst;
      }
    }
  }

  // Fix up references in the cloned instructions
  for (BasicBlock &BB : *NewF) {
    for (Instruction &I : BB) {
      for (unsigned i = 0; i < I.getNumOperands(); ++i) {
        Value *Op = I.getOperand(i);
        if (VMap.count(Op)) {
          I.setOperand(i, VMap[Op]);
        }
      }
    }
  }

  // Create a function analysis manager
  FunctionAnalysisManager FAM;

  // Register the analysis passes
  FAM.registerPass([&] { return TensorDataFlowAnalysis(); });
  FAM.registerPass([&] { return TensorAccessPatternAnalysis(); });

  // Create an auto-tuner with a fixed strategy
  AutoTuner Tuner;

  // Apply the optimization strategy
  Tuner.applyStrategy(*NewF, FAM, Strategy);

  return NewF;
}

double Profiler::measureExecutionTime(Function &F, int NumRuns) {
  // Create an execution engine
  Module *M = F.getParent();
  if (!M) {
    return 0.0;
  }

  // Create a new module for execution to avoid modifying the original
  std::unique_ptr<Module> ClonedModule = std::make_unique<Module>(M->getName(), M->getContext());

  // Copy the function to the new module
  Function *ClonedF = Function::Create(
    F.getFunctionType(),
    F.getLinkage(),
    F.getName(),
    ClonedModule.get()
  );

  // Copy the function body using ValueToValueMapTy
  ValueToValueMapTy VMap;
  for (const BasicBlock &BB : F) {
    BasicBlock *NewBB = BasicBlock::Create(F.getContext(), BB.getName(), ClonedF);
    VMap[&BB] = NewBB;

    for (const Instruction &I : BB) {
      Instruction *NewInst = I.clone();
      if (NewInst->getType()->isVoidTy()) {
        NewBB->getInstList().push_back(NewInst);
      } else {
        NewBB->getInstList().push_back(NewInst);
        VMap[&I] = NewInst;
      }
    }
  }

  // Fix up references
  for (BasicBlock &BB : *ClonedF) {
    for (Instruction &I : BB) {
      for (unsigned i = 0; i < I.getNumOperands(); ++i) {
        Value *Op = I.getOperand(i);
        if (VMap.count(Op)) {
          I.setOperand(i, VMap[Op]);
        }
      }
    }
  }

  // Create the execution engine
  std::string ErrorMsg;
  EngineBuilder builder(std::move(ClonedModule));
  builder.setErrorStr(&ErrorMsg);
  builder.setEngineKind(EngineKind::JIT);

  std::unique_ptr<ExecutionEngine> EE(builder.create());

  if (!EE) {
    LLVM_DEBUG(dbgs() << "Profiler: Failed to create execution engine: " << ErrorMsg << "\n");
    return 0.0;
  }

  // Measure the execution time
  auto Start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NumRuns; ++i) {
    // Run the function
    std::vector<GenericValue> Args;
    EE->runFunction(ClonedF, Args);
  }

  auto End = std::chrono::high_resolution_clock::now();

  // Calculate the average execution time in milliseconds
  double ExecutionTime = std::chrono::duration_cast<std::chrono::microseconds>(End - Start).count() / 1000.0 / NumRuns;

  return ExecutionTime;
}
