package torch
package nn
package functional

import Derive.derive
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  BatchNormFuncOptions,
  TensorOptional,
  GroupNormFuncOptions,
  InstanceNormFuncOptions,
  LayerNormFuncOptions,
  LocalResponseNormOptions,
  NormalizeFuncOptions,
  TensorVector,
  ScalarTypeOptional
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}

private[torch] trait BatchNorm {

  def batch_norm[D <: DType](
      input: Tensor[D],
      running_mean: Tensor[D],
      running_var: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D],
      training: Boolean = false,
      momentum: Double = 0.1,
      eps: Double = 1e-05
  ): Tensor[D] = {
    val options: BatchNormFuncOptions = new BatchNormFuncOptions()
    options.momentum().put(momentum)
    options.eps().put(eps)
    options.weight.put(weight.native)
    options.bias().put(bias.native)
    options.training().put(training)
    val result =
      torchNative.batch_norm(input.native, running_mean.native, running_var.native, options)
    fromNative(result)
  }

  def instance_norm[D <: DType](
      input: Tensor[D],
      running_mean: Tensor[D],
      running_var: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D],
      training: Boolean = false,
      momentum: Double = 0.1,
      eps: Double = 1e-05,
      use_input_stats: Boolean = false
  ): Tensor[D] = {
    val options: InstanceNormFuncOptions = new InstanceNormFuncOptions()
    options.momentum().put(momentum)
    options.eps().put(eps)
    options.weight.put(weight.native)
    options.bias().put(bias.native)
    options.use_input_stats().put(use_input_stats)
    options.running_mean().put(running_mean.native)
    options.running_var().put(running_var.native)
    val result = torchNative.instance_norm(input.native, options)
    fromNative(result)
  }

  def group_norm[D <: DType](
      input: Tensor[D],
      numGroups: Long,
      numChannels: Long,
      weight: Tensor[D],
      bias: Tensor[D],
      training: Boolean = false,
      eps: Double = 1e-05,
      affine: Boolean
  ): Tensor[D] = {
    val options: GroupNormFuncOptions = new GroupNormFuncOptions(numGroups)
//    options.momentum().put(0.1)
    options.eps().put(eps)
    options.weight().put(weight.native)
    options.bias().put(bias.native)
//    options.num_groups().put(numGroups)
//    options.num_channels().put(numChannels)
    val result = torchNative.group_norm(input.native, options)
    fromNative(result)
  }

  def layer_norm[D <: DType](
      input: Tensor[D],
      normalized_shape: Tensor[D],
      weight: Tensor[D],
      bias: Tensor[D],
      eps: Double = 1e-05
  ): Tensor[D] = {
    val tensorVector = new TensorVector(normalized_shape.native)
    val options: LayerNormFuncOptions = new LayerNormFuncOptions(tensorVector)

    options.normalized_shape().put(tensorVector)
    options.eps().put(eps)
    options.weight().put(weight.native)
    options.bias().put(bias.native)
    val result = torchNative.layer_norm(input.native, options)
    fromNative(result)
  }

  def local_response_norm[D <: DType](
      input: Tensor[D],
      size: Long,
      alpha: Double = 0.0001,
      beta: Double = 0.75,
      k: Double = 0.1
  ): Tensor[D] = {
    val options = new LocalResponseNormOptions(size)
    options.alpha().put(alpha)
    options.beta().put(beta)
    options.k().put(k)
    val result = torchNative.local_response_norm(input.native, options)
    fromNative(result)
  }

  def normalize[D <: DType](
      input: Tensor[D],
      p: Double = 2.0,
      dim: Long = 1,
      eps: Double = 1e-12,
      out: Tensor[D]
  ): Tensor[D] = {
    val options: NormalizeFuncOptions = new NormalizeFuncOptions()
    options.dim().put(dim)
    options.eps().put(eps)
    options.p().put(p)
    options.out().put(TensorOptional(out.native))
    val result = torchNative.normalize(input.native, options)
    fromNative(result)
  }

}

//public native
//@ByRef @NoException(true) Tensor weight();
//public native
//@ByRef @NoException(true) Tensor bias();
//public native
//@Cast("bool*") @ByRef @NoException(true) BoolPointer training();
//public native
//@ByRef @NoException(true) DoubleOptional momentum();
//public native
//@ByRef @NoException(true) DoublePointer eps();

//  @Namespace("torch::nn::functional")  125
//  @Namespace("torch::nn::init") 28
// @Namespace("torch::special") 136
//  @Namespace("torch::nn::functional") public static native @ByVal Tensor batch_norm(
//  @Const @ByRef Tensor input,
//  @Const @ByRef Tensor running_mean,
//  @Const @ByRef Tensor running_var,
//  @Const @ByRef(nullValue = "torch::nn::functional::BatchNormFuncOptions{}") BatchNormFuncOptions options);
//  @Namespace("torch::nn::functional") public static native @ByVal Tensor batch_norm(
//  @Const @ByRef Tensor input,
//  @Const @ByRef Tensor running_mean,
//  @Const @ByRef Tensor running_var);

//  @Namespace("torch::linalg") public static native @ByVal Tensor norm(
//  @Const @ByRef Tensor self,
//  @Const @ByRef ScalarOptional opt_ord,
//  @ByVal LongArrayRefOptional opt_dim,
//  @Cast("bool") boolean keepdim,
//  @ByVal ScalarTypeOptional opt_dtype);
//  @Namespace("torch::linalg") public static native @ByVal Tensor norm(
//  @Const @ByRef Tensor self,
//  @Const @ByRef ScalarOptional opt_ord,
//  @ByVal @Cast({"int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"}) @StdVector long[] opt_dim,
//  @Cast("bool") boolean keepdim,
//  @ByVal ScalarTypeOptional opt_dtype);
//
//  @Namespace("torch::linalg") public static native @ByVal Tensor norm(
//  @Const @ByRef Tensor self,
//  @StdString BytePointer ord,
//  @ByVal LongArrayRefOptional opt_dim,
//  @Cast("bool") boolean keepdim,
//  @ByVal ScalarTypeOptional opt_dtype);
//  @Namespace("torch::linalg") public static native @ByVal Tensor norm(
//  @Const @ByRef Tensor self,
//  @StdString String ord,
//  @ByVal @Cast({"int64_t*", "c10::ArrayRef<int64_t>", "std::vector<int64_t>&"}) @StdVector long[] opt_dim,
//  @Cast("bool") boolean keepdim,
//  @ByVal ScalarTypeOptional opt_dtype);
