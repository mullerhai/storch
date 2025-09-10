//package torch.nn.loss //BCELossImpl
package torch
package nn
package loss

import org.bytedeco.pytorch.BCELossImpl
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

final class BCELoss extends LossFunc {
  override private[torch] val nativeModule: BCELossImpl = BCELossImpl()

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native) // .output()
    )
  }

  def forward[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native) // .output()
    )
  }

  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )
}
object BCELoss {
  def apply(): BCELoss = new BCELoss()
}
