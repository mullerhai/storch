//package torch.nn.functional
//
//import org.bytedeco.javacpp.*
//import org.bytedeco.javacpp.annotation.*
//import torch.DType
//import org.bytedeco.pytorch
//import org.bytedeco.pytorch.global.torch as torchNative
//import torch.internal.NativeConverters.fromNative
//import org.bytedeco.pytorch.ScalarTypeOptional
//import java.nio.{DoubleBuffer, LongBuffer}
//import scala.reflect.ClassTag
//
//@Platform(include = Array( "ATen/NativeFunctions.h", "adapters/OptionalAdapter.h", "adapters/StdArrayAdapter.h" ))
//@NoOffset private[torch] trait NavFunc {
//
//  @native
//  @Namespace("at::native")
//  @ByVal
//  def relu[TT <: DType](@Const @ByRef input: Tensor[TT]): Tensor[TT] //= fromNative(torchNative.relu(input.native))
////
//}
