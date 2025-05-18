//package torch.nn.loss
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.TripletMarginLossImpl

final class TripletMarginLoss extends LossFunc {
  override private[torch] val nativeModule: TripletMarginLossImpl = TripletMarginLossImpl()

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()
  
  def apply[D <: DType](anchor: Tensor[D], positive: Tensor[D], negative: Tensor[?]): Tensor[D] =
    fromNative(
      nativeModule.forward(anchor.native, positive.native, negative.native)
    )

  override def apply[D <: DType](inputs: Tensor[D]*)(negative: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    val positive = inputs.toSeq.last

    fromNative(
      nativeModule.forward(input.native, positive.native, negative.native) // .output()
    )
  }

}

object TripletMarginLoss {
  
  def apply() : TripletMarginLoss = new TripletMarginLoss()
}
