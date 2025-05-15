#include "gtest/gtest.h"

#include "Analysis/TensorDataFlowAnalysis.h"
#include "Analysis/TensorAccessPatternAnalysis.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;
using namespace llvm::tensor;

// Test fixture for analysis tests
class AnalysisTest : public ::testing::Test {
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
  
  // Helper function to create a simple function with tensor operations
  Function* createTestFunction() {
    // Create function type
    FunctionType* FuncType = FunctionType::get(Type::getVoidTy(*Context), false);
    
    // Create function
    Function* F = Function::Create(FuncType, Function::ExternalLinkage, "test_function", M.get());
    
    // Create basic block
    BasicBlock* BB = BasicBlock::Create(*Context, "entry", F);
    
    // Create IR builder
    IRBuilder<> Builder(BB);
    
    // Create tensor operations
    // For simplicity, we'll just create some calls to functions with "tensor" in the name
    
    // Create function types for tensor operations
    FunctionType* TensorOpType = FunctionType::get(Type::getFloatTy(*Context), {Type::getFloatPtrTy(*Context), Type::getFloatPtrTy(*Context), Type::getFloatPtrTy(*Context)}, false);
    
    // Create tensor operation functions
    Function* TensorAddFunc = Function::Create(TensorOpType, Function::ExternalLinkage, "tensor_add", M.get());
    Function* TensorMulFunc = Function::Create(TensorOpType, Function::ExternalLinkage, "tensor_mul", M.get());
    
    // Create allocas for tensor data
    AllocaInst* A = Builder.CreateAlloca(Type::getFloatTy(*Context), nullptr, "A");
    AllocaInst* B = Builder.CreateAlloca(Type::getFloatTy(*Context), nullptr, "B");
    AllocaInst* C = Builder.CreateAlloca(Type::getFloatTy(*Context), nullptr, "C");
    AllocaInst* D = Builder.CreateAlloca(Type::getFloatTy(*Context), nullptr, "D");
    
    // Create tensor operations
    CallInst* AddCall = Builder.CreateCall(TensorAddFunc, {A, B, C}, "tensor_add_call");
    CallInst* MulCall = Builder.CreateCall(TensorMulFunc, {C, D, A}, "tensor_mul_call");
    
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

// Test the tensor data flow analysis
TEST_F(AnalysisTest, TensorDataFlowAnalysisTest) {
  // Create a test function
  Function* F = createTestFunction();
  
  // Run the analysis
  auto& TDFA = FAM.getResult<TensorDataFlowAnalysis>(*F);
  
  // Check that the analysis identified tensor operations
  EXPECT_GT(TDFA.size(), 0);
  
  // Additional checks could be added here to verify the data flow graph
}

// Test the tensor access pattern analysis
TEST_F(AnalysisTest, TensorAccessPatternAnalysisTest) {
  // Create a test function
  Function* F = createTestFunction();
  
  // Run the analysis
  auto& TAPA = FAM.getResult<TensorAccessPatternAnalysis>(*F);
  
  // Check that the analysis identified tensor operations
  EXPECT_GT(TAPA.size(), 0);
  
  // Additional checks could be added here to verify the access patterns
}

// Test both analyses together
TEST_F(AnalysisTest, BothAnalysesTest) {
  // Create a test function
  Function* F = createTestFunction();
  
  // Run the analyses
  auto& TDFA = FAM.getResult<TensorDataFlowAnalysis>(*F);
  auto& TAPA = FAM.getResult<TensorAccessPatternAnalysis>(*F);
  
  // Check that both analyses identified tensor operations
  EXPECT_GT(TDFA.size(), 0);
  EXPECT_GT(TAPA.size(), 0);
  
  // Additional checks could be added here to verify the results of both analyses
}
