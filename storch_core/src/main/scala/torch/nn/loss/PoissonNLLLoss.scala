//package torch.nn.loss
package torch
package nn
package loss

import org.bytedeco.pytorch.PoissonNLLLossImpl
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

final class PoissonNLLLoss extends LossFunc {
  override private[torch] val nativeModule: PoissonNLLLossImpl = PoissonNLLLossImpl()

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )

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

}

object PoissonNLLLoss {

  def apply(): PoissonNLLLoss = new PoissonNLLLoss()
}
