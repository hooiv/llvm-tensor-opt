from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# Get the path to the LLVM Tensor Optimization library
LLVM_TENSOR_OPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLVM_TENSOR_OPT_BUILD_DIR = os.path.join(LLVM_TENSOR_OPT_DIR, 'build')

# Define the extension module
setup(
    name='llvm_tensor_opt_pytorch',
    ext_modules=[
        cpp_extension.CppExtension(
            name='llvm_tensor_opt_pytorch',
            sources=[
                'llvm_tensor_opt_pytorch.cpp',
                'tensor_compiler.cpp',
                'tensor_jit.cpp',
            ],
            include_dirs=[
                os.path.join(LLVM_TENSOR_OPT_DIR, 'include'),
                # Add LLVM include directories
                '/usr/include/llvm',
                '/usr/include/llvm-c',
            ],
            library_dirs=[
                os.path.join(LLVM_TENSOR_OPT_BUILD_DIR, 'src/Transforms'),
                os.path.join(LLVM_TENSOR_OPT_BUILD_DIR, 'src/Analysis'),
                os.path.join(LLVM_TENSOR_OPT_BUILD_DIR, 'src/CUDA'),
                # Add LLVM library directories
                '/usr/lib/llvm/lib',
            ],
            libraries=[
                'LLVMTensorTransforms',
                'LLVMTensorAnalysis',
                'LLVMTensorCUDA',
                # Add LLVM libraries
                'LLVM',
            ],
            extra_compile_args=['-std=c++17'],
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
