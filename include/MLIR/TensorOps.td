#ifndef TENSOR_OPS
#define TENSOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TensorOptDialect : Dialect {
  let name = "tensor_opt";
  let cppNamespace = "::mlir::tensor_opt";
}

class TensorOptOp<string mnemonic, list<Trait> traits = []> :
    Op<TensorOptDialect, mnemonic, traits>;

def MatMulOp : TensorOptOp<"matmul", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication on two input tensors.
    
    Example:
    ```mlir
    %c = tensor_opt.matmul %a, %b : tensor<MxK>, tensor<KxN> -> tensor<MxN>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$lhs,
    RankedTensorType:$rhs
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def ConvOp : TensorOptOp<"conv", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Convolution operation";
  let description = [{
    Performs convolution on input tensor with a kernel.
    
    Example:
    ```mlir
    %c = tensor_opt.conv %input, %kernel : tensor<NxCxHxW>, tensor<KxCxKHxKW> -> tensor<NxKxOHxOW>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$input,
    RankedTensorType:$kernel,
    OptionalAttr<I64ArrayAttr>:$strides,
    OptionalAttr<I64ArrayAttr>:$padding
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $input `,` $kernel attr-dict `:` type($input) `,` type($kernel) `->` type($result)
  }];
}

def ElementWiseAddOp : TensorOptOp<"add", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition on two input tensors.
    
    Example:
    ```mlir
    %c = tensor_opt.add %a, %b : tensor<NxM>, tensor<NxM> -> tensor<NxM>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$lhs,
    RankedTensorType:$rhs
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def ElementWiseMulOp : TensorOptOp<"mul", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Element-wise multiplication operation";
  let description = [{
    Performs element-wise multiplication on two input tensors.
    
    Example:
    ```mlir
    %c = tensor_opt.mul %a, %b : tensor<NxM>, tensor<NxM> -> tensor<NxM>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$lhs,
    RankedTensorType:$rhs
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def TransposeOp : TensorOptOp<"transpose", [NoSideEffect]> {
  let summary = "Transpose operation";
  let description = [{
    Performs transpose on input tensor.
    
    Example:
    ```mlir
    %b = tensor_opt.transpose %a : tensor<MxN> -> tensor<NxM>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$input,
    OptionalAttr<I64ArrayAttr>:$permutation
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def ReshapeOp : TensorOptOp<"reshape", [NoSideEffect]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes input tensor to a new shape.
    
    Example:
    ```mlir
    %b = tensor_opt.reshape %a : tensor<MxN> -> tensor<M*N>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$input,
    OptionalAttr<I64ArrayAttr>:$new_shape
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def SoftmaxOp : TensorOptOp<"softmax", [NoSideEffect]> {
  let summary = "Softmax operation";
  let description = [{
    Performs softmax on input tensor.
    
    Example:
    ```mlir
    %b = tensor_opt.softmax %a : tensor<NxM> -> tensor<NxM>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$input,
    OptionalAttr<I64Attr>:$axis
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def AttentionOp : TensorOptOp<"attention", [NoSideEffect]> {
  let summary = "Attention operation";
  let description = [{
    Performs attention operation on query, key, and value tensors.
    
    Example:
    ```mlir
    %output = tensor_opt.attention %query, %key, %value : tensor<BxNxH>, tensor<BxMxH>, tensor<BxMxH> -> tensor<BxNxH>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$query,
    RankedTensorType:$key,
    RankedTensorType:$value,
    OptionalAttr<F32Attr>:$scale
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $query `,` $key `,` $value attr-dict `:` type($query) `,` type($key) `,` type($value) `->` type($result)
  }];
}

def LayerNormOp : TensorOptOp<"layer_norm", [NoSideEffect]> {
  let summary = "Layer normalization operation";
  let description = [{
    Performs layer normalization on input tensor.
    
    Example:
    ```mlir
    %b = tensor_opt.layer_norm %a, %gamma, %beta : tensor<BxN>, tensor<N>, tensor<N> -> tensor<BxN>
    ```
  }];
  
  let arguments = (ins
    RankedTensorType:$input,
    RankedTensorType:$gamma,
    RankedTensorType:$beta,
    OptionalAttr<F32Attr>:$epsilon
  );
  
  let results = (outs
    RankedTensorType:$result
  );
  
  let assemblyFormat = [{
    $input `,` $gamma `,` $beta attr-dict `:` type($input) `,` type($gamma) `,` type($beta) `->` type($result)
  }];
}

#endif // TENSOR_OPS
