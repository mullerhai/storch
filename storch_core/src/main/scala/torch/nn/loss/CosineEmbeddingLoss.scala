//package torch.nn.loss
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.CosineEmbeddingLossImpl

final class CosineEmbeddingLoss extends LossFunc {
  override private[torch] val nativeModule: CosineEmbeddingLossImpl = CosineEmbeddingLossImpl()

  override def hasBias(): Boolean = false

  def apply[D <: DType](input1: Tensor[D], input2: Tensor[D], target: Tensor[?]): Tensor[D] =
    fromNative(
      nativeModule.forward(input1.native, input2.native, target.native)
    )

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {

    val input1 = inputs.toSeq.head
    val input2 = inputs.toSeq.last
    fromNative(
      nativeModule.forward(input1.native, input2.native, target.native) // .output()
    )
  }
}
object CosineEmbeddingLoss {
  def apply() : CosineEmbeddingLoss = new CosineEmbeddingLoss()
}