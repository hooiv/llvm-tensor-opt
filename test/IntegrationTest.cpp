#include "gtest/gtest.h"

#include "Transforms/TensorFusion.h"
#include "Transforms/TensorVectorization.h"
#include "Transforms/TensorParallelization.h"
#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorAccessPatternAnalysis.h"
#include "CUDA/CUDAOffloader.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::tensor;

// Test fixture for integration tests
class IntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a new context and module for each test
    Context = std::make_unique<LLVMContext>();
    M = std::make_unique<Module>("test_module", *Context);
    
    // Create a pass builder and analysis managers
    PassBuilder PB;
    
    // Register the analysis passes
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    
    // Register our custom analysis passes
    FAM.registerPass([&] { return TensorDataFlowAnalysis(); });
    FAM.registerPass([&] { return TensorAccessPatternAnalysis(); });
  }
  
  // Helper function to create a matrix multiplication function
  Function* createMatrixMultiplyFunction() {
    // Create function type
    FunctionType* FuncType = FunctionType::get(Type::getVoidTy(*Context), {
      Type::getFloatPtrTy(*Context), // A
      Type::getFloatPtrTy(*Context), // B
      Type::getFloatPtrTy(*Context), // C
      Type::getInt32Ty(*Context),    // M
      Type::getInt32Ty(*Context),    // N
      Type::getInt32Ty(*Context)     // K
    }, false);
    
    // Create function
    Function* F = Function::Create(FuncType, Function::ExternalLinkage, "matrix_multiply", M.get());
    
    // Name the function arguments
    auto ArgIt = F->arg_begin();
    Argument* A = &*ArgIt++;
    A->setName("A");
    Argument* B = &*ArgIt++;
    B->setName("B");
    Argument* C = &*ArgIt++;
    C->setName("C");
    Argument* M = &*ArgIt++;
    M->setName("M");
    Argument* N = &*ArgIt++;
    N->setName("N");
    Argument* K = &*ArgIt++;
    K->setName("K");
    
    // Create basic block
    BasicBlock* BB = BasicBlock::Create(*Context, "entry", F);
    
    // Create IR builder
    IRBuilder<> Builder(BB);
    
    // Create nested loops for matrix multiplication
    // for (int i = 0; i < M; ++i)
    //   for (int j = 0; j < N; ++j)
    //     for (int k = 0; k < K; ++k)
    //       C[i*N + j] += A[i*K + k] * B[k*N + j];
    
    // Create loop variables
    AllocaInst* IVar = Builder.CreateAlloca(Type::getInt32Ty(*Context), nullptr, "i");
    AllocaInst* JVar = Builder.CreateAlloca(Type::getInt32Ty(*Context), nullptr, "j");
    AllocaInst* KVar = Builder.CreateAlloca(Type::getInt32Ty(*Context), nullptr, "k");
    
    // Initialize i = 0
    Builder.CreateStore(Builder.getInt32(0), IVar);
    
    // Create outer loop header
    BasicBlock* OuterLoopHeader = BasicBlock::Create(*Context, "outer_loop_header", F);
    BasicBlock* OuterLoopBody = BasicBlock::Create(*Context, "outer_loop_body", F);
    BasicBlock* OuterLoopExit = BasicBlock::Create(*Context, "outer_loop_exit", F);
    
    // Branch to outer loop header
    Builder.CreateBr(OuterLoopHeader);
    
    // Outer loop header
    Builder.SetInsertPoint(OuterLoopHeader);
    Value* IVal = Builder.CreateLoad(Type::getInt32Ty(*Context), IVar, "i_val");
    Value* OuterCond = Builder.CreateICmpSLT(IVal, M, "outer_cond");
    Builder.CreateCondBr(OuterCond, OuterLoopBody, OuterLoopExit);
    
    // Outer loop body
    Builder.SetInsertPoint(OuterLoopBody);
    
    // Initialize j = 0
    Builder.CreateStore(Builder.getInt32(0), JVar);
    
    // Create middle loop header
    BasicBlock* MiddleLoopHeader = BasicBlock::Create(*Context, "middle_loop_header", F);
    BasicBlock* MiddleLoopBody = BasicBlock::Create(*Context, "middle_loop_body", F);
    BasicBlock* MiddleLoopExit = BasicBlock::Create(*Context, "middle_loop_exit", F);
    
    // Branch to middle loop header
    Builder.CreateBr(MiddleLoopHeader);
    
    // Middle loop header
    Builder.SetInsertPoint(MiddleLoopHeader);
    Value* JVal = Builder.CreateLoad(Type::getInt32Ty(*Context), JVar, "j_val");
    Value* MiddleCond = Builder.CreateICmpSLT(JVal, N, "middle_cond");
    Builder.CreateCondBr(MiddleCond, MiddleLoopBody, MiddleLoopExit);
    
    // Middle loop body
    Builder.SetInsertPoint(MiddleLoopBody);
    
    // Initialize k = 0
    Builder.CreateStore(Builder.getInt32(0), KVar);
    
    // Create inner loop header
    BasicBlock* InnerLoopHeader = BasicBlock::Create(*Context, "inner_loop_header", F);
    BasicBlock* InnerLoopBody = BasicBlock::Create(*Context, "inner_loop_body", F);
    BasicBlock* InnerLoopExit = BasicBlock::Create(*Context, "inner_loop_exit", F);
    
    // Branch to inner loop header
    Builder.CreateBr(InnerLoopHeader);
    
    // Inner loop header
    Builder.SetInsertPoint(InnerLoopHeader);
    Value* KVal = Builder.CreateLoad(Type::getInt32Ty(*Context), KVar, "k_val");
    Value* InnerCond = Builder.CreateICmpSLT(KVal, K, "inner_cond");
    Builder.CreateCondBr(InnerCond, InnerLoopBody, InnerLoopExit);
    
    // Inner loop body
    Builder.SetInsertPoint(InnerLoopBody);
    
    // C[i*N + j] += A[i*K + k] * B[k*N + j]
    
    // Calculate indices
    Value* AIdx = Builder.CreateAdd(
      Builder.CreateMul(IVal, K),
      KVal
    );
    Value* BIdx = Builder.CreateAdd(
      Builder.CreateMul(KVal, N),
      JVal
    );
    Value* CIdx = Builder.CreateAdd(
      Builder.CreateMul(IVal, N),
      JVal
    );
    
    // Load values
    Value* AVal = Builder.CreateLoad(Type::getFloatTy(*Context), Builder.CreateGEP(Type::getFloatTy(*Context), A, AIdx), "a_val");
    Value* BVal = Builder.CreateLoad(Type::getFloatTy(*Context), Builder.CreateGEP(Type::getFloatTy(*Context), B, BIdx), "b_val");
    Value* CVal = Builder.CreateLoad(Type::getFloatTy(*Context), Builder.CreateGEP(Type::getFloatTy(*Context), C, CIdx), "c_val");
    
    // Perform multiplication and addition
    Value* Mul = Builder.CreateFMul(AVal, BVal, "mul");
    Value* Add = Builder.CreateFAdd(CVal, Mul, "add");
    
    // Store result
    Builder.CreateStore(Add, Builder.CreateGEP(Type::getFloatTy(*Context), C, CIdx));
    
    // Increment k
    Value* KNext = Builder.CreateAdd(KVal, Builder.getInt32(1), "k_next");
    Builder.CreateStore(KNext, KVar);
    
    // Branch back to inner loop header
    Builder.CreateBr(InnerLoopHeader);
    
    // Inner loop exit
    Builder.SetInsertPoint(InnerLoopExit);
    
    // Increment j
    Value* JNext = Builder.CreateAdd(JVal, Builder.getInt32(1), "j_next");
    Builder.CreateStore(JNext, JVar);
    
    // Branch back to middle loop header
    Builder.CreateBr(MiddleLoopHeader);
    
    // Middle loop exit
    Builder.SetInsertPoint(MiddleLoopExit);
    
    // Increment i
    Value* INext = Builder.CreateAdd(IVal, Builder.getInt32(1), "i_next");
    Builder.CreateStore(INext, IVar);
    
    // Branch back to outer loop header
    Builder.CreateBr(OuterLoopHeader);
    
    // Outer loop exit
    Builder.SetInsertPoint(OuterLoopExit);
    
    // Create return
    Builder.CreateRetVoid();
    
    // Verify the function
    EXPECT_FALSE(verifyFunction(*F, &errs()));
    
    return F;
  }
  
  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<Module> M;
  
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
};

// Test the full optimization pipeline on matrix multiplication
TEST_F(IntegrationTest, MatrixMultiplyOptimizationTest) {
  // Create a matrix multiplication function
  Function* F = createMatrixMultiplyFunction();
  
  // Create a function pass manager
  FunctionPassManager FPM;
  
  // Add our custom optimization passes
  FPM.addPass(TensorFusionPass());
  FPM.addPass(TensorVectorizationPass());
  FPM.addPass(TensorParallelizationPass());
  
  // Run the passes
  PreservedAnalyses PA = FPM.run(*F, FAM);
  
  // Verify the function after optimization
  EXPECT_FALSE(verifyFunction(*F, &errs()));
  
  // Offload to CUDA
  cuda::CUDAOffloader Offloader;
  Offloader.offloadFunction(*F);
  
  // Verify the function after CUDA offloading
  EXPECT_FALSE(verifyFunction(*F, &errs()));
  
  // Additional checks could be added here to verify the optimized function
}
