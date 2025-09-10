package torch
package nn
package modules
package container
import org.bytedeco.pytorch
import sourcecode.Name
import scala.collection.mutable.ListBuffer
import org.bytedeco.pytorch.Module
import torch.internal.NativeConverters.fromNative

object Parameter {

  def apply[D <: DType](
      name: String,
      weight: Tensor[D],
      requires_grad: Boolean = true
  )(using nativeModule: Module = pytorch.Module()): Tensor[D] =
    nativeModule.register_parameter(name, weight.native, requires_grad)
    weight
  def register_parameter[D <: DType](
      name: String,
      weight: Tensor[D],
      requires_grad: Boolean = true
  )(using nativeModule: Module = pytorch.Module()): Tensor[D] =
    nativeModule.register_parameter(name, weight.native, requires_grad)
    weight
}
