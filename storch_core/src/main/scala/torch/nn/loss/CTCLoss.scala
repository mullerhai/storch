//package torch.nn.loss
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.CTCLossImpl

final class CTCLoss extends LossFunc {
  override private[torch] val nativeModule: CTCLossImpl = CTCLossImpl()

  override def hasBias(): Boolean = false

  def apply[D <: DType](
      input: Tensor[D],
      target: Tensor[?],
      w: Tensor[?],
      k: Tensor[D]
  ): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native, w.native, k.native)
  )

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    val k = inputs.toSeq.last
    val w = inputs.toSeq.dropRight(1).head
    fromNative(
      nativeModule.forward(input.native, target.native, w.native, k.native) // .output()
    )
  }
}

object CTCLoss {
  def apply() : CTCLoss = new CTCLoss()

}