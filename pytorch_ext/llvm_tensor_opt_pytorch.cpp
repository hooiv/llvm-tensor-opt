#include <torch/extension.h>
#include "tensor_compiler.h"
#include "tensor_jit.h"

// Register the tensor optimization passes with PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PyTorch extension for LLVM Tensor Optimization";
  
  // Register the tensor compiler
  py::class_<TensorCompiler>(m, "TensorCompiler")
    .def(py::init<>())
    .def("compile", &TensorCompiler::compile)
    .def("optimize", &TensorCompiler::optimize)
    .def("get_ir", &TensorCompiler::getIR);
  
  // Register the tensor JIT
  py::class_<TensorJIT>(m, "TensorJIT")
    .def(py::init<>())
    .def("add_module", &TensorJIT::addModule)
    .def("optimize_module", &TensorJIT::optimizeModule)
    .def("run", &TensorJIT::run);
  
  // Register optimization flags
  m.def("enable_fusion", &enableFusion);
  m.def("enable_vectorization", &enableVectorization);
  m.def("enable_parallelization", &enableParallelization);
  m.def("enable_cuda_offload", &enableCUDAOffload);
}
