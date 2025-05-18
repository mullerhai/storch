//package torch.nn.loss
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.HingeEmbeddingLossImpl

final class HingeEmbeddingLoss extends LossFunc {
  override private[torch] val nativeModule: HingeEmbeddingLossImpl = HingeEmbeddingLossImpl()

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()
  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native) // .output()
    )
  }
}

object HingeEmbeddingLoss {

  def apply(): HingeEmbeddingLoss = new HingeEmbeddingLoss()
}