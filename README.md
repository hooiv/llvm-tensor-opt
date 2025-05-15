# LLVM Tensor Optimization Pass

This project implements a set of LLVM optimization passes specifically designed for tensor operations, with a focus on GPU execution. The passes analyze tensor operations in LLVM IR and apply various optimizations to improve performance on GPU architectures.

## Features

- **Tensor Fusion**: Fuses multiple tensor operations to reduce memory traffic and improve GPU utilization
- **Tensor Vectorization**: Vectorizes tensor operations to exploit SIMD execution on GPUs
- **Tensor Parallelization**: Transforms tensor operations to exploit GPU parallelism
- **CUDA Offloading**: Offloads tensor operations to CUDA for execution on NVIDIA GPUs
- **Analysis Passes**: Provides analysis passes to identify tensor operations and their access patterns

## Requirements

- LLVM 13.0 or later
- CMake 3.13.4 or later
- C++17 compatible compiler
- CUDA Toolkit 11.0 or later (optional, for GPU support)
- Google Test (optional, for running tests)

## Building

```bash
# Clone the repository
git clone https://github.com/hooiv/llvm-tensor-opt.git
cd llvm-tensor-opt

# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build .

# Run tests
ctest
```

## Usage

### Using the Optimization Tool

The project provides a command-line tool to apply the optimization passes to LLVM IR files:

```bash
# Apply all optimizations
./tensor_opt_tool input.ll -o output.ll --fusion --vectorization --parallelization --cuda-offload

# Apply specific optimizations
./tensor_opt_tool input.ll -o output.ll --fusion --vectorization
```

### Using the Optimization Passes in Your LLVM Project

You can integrate the optimization passes into your LLVM-based project:

```cpp
#include "Transforms/TensorFusion.h"
#include "Transforms/TensorVectorization.h"
#include "Transforms/TensorParallelization.h"

// Create a function pass manager
llvm::FunctionPassManager FPM;

// Add the optimization passes
FPM.addPass(llvm::tensor::TensorFusionPass());
FPM.addPass(llvm::tensor::TensorVectorizationPass());
FPM.addPass(llvm::tensor::TensorParallelizationPass());

// Run the passes on a function
FPM.run(Function, FAM);
```

## Examples

The project includes several examples to demonstrate the optimization passes:

- **Matrix Multiplication**: A simple matrix multiplication example that demonstrates the performance improvements from the optimization passes
- **Tensor Benchmark**: A benchmark tool to measure the performance of tensor operations with and without the optimization passes

## Project Structure

- `include/`: Header files
  - `Transforms/`: Optimization pass headers
  - `Analysis/`: Analysis pass headers
  - `CUDA/`: CUDA integration headers
- `src/`: Source files
  - `Transforms/`: Optimization pass implementations
  - `Analysis/`: Analysis pass implementations
  - `CUDA/`: CUDA integration implementations
- `test/`: Test files
- `examples/`: Example applications
- `docs/`: Documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The LLVM Project (https://llvm.org/)
- NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
