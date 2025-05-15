import torch
import llvm_tensor_opt_pytorch as llvm_opt

def main():
    # Create a tensor compiler
    compiler = llvm_opt.TensorCompiler()
    
    # Create input tensors
    a = torch.randn(1024, 1024, dtype=torch.float32)
    b = torch.randn(1024, 1024, dtype=torch.float32)
    
    # Compile a matrix multiplication operation
    compiler.compile(a, "matmul")
    
    # Optimize the compiled operation
    compiler.optimize()
    
    # Get the optimized IR
    ir = compiler.get_ir()
    print(f"Generated IR:\n{ir}")
    
    # Create a tensor JIT
    jit = llvm_opt.TensorJIT()
    
    # Add the module to the JIT
    jit.add_module(ir)
    
    # Optimize the module
    jit.optimize_module("pytorch_module")
    
    # Run the function
    inputs = [a, b]
    output = jit.run("tensor_matmul", inputs)
    
    # Compare with PyTorch's implementation
    torch_output = torch.matmul(a, b)
    
    # Check if the results are close
    if torch.allclose(output, torch_output, rtol=1e-5, atol=1e-5):
        print("Results match!")
    else:
        print("Results don't match!")
        print(f"Max difference: {torch.max(torch.abs(output - torch_output))}")

if __name__ == "__main__":
    # Enable optimizations
    llvm_opt.enable_fusion(True)
    llvm_opt.enable_vectorization(True)
    llvm_opt.enable_parallelization(True)
    llvm_opt.enable_cuda_offload(torch.cuda.is_available())
    
    main()
