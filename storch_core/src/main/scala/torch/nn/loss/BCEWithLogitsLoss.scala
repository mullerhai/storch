//package torch.nn.loss //BCEWithLogitsLossImpl
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.BCEWithLogitsLossImpl

final class BCEWithLogitsLoss extends LossFunc {
  override private[torch] val nativeModule: BCEWithLogitsLossImpl = BCEWithLogitsLossImpl()

  override def hasBias(): Boolean = false
  def weight[D <: DType](): Tensor[D] = fromNative(nativeModule.weight())
  def pos_weight[D <: DType](): Tensor[D] = fromNative(nativeModule.pos_weight())

//  def weight(weight: Tensor[D]) = nativeModule.weight(weight.native)

  def reset(): Unit = nativeModule.reset()
  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )
  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

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
object BCEWithLogitsLoss {
  def apply(): BCEWithLogitsLoss = new BCEWithLogitsLoss()
}

//  public native @ByRef Tensor weight(); public native BCEWithLogitsLossImpl weight(Tensor setter);
//
//  /** A weight of positive examples. */
//  public native @ByRef Tensor pos_weight(); public native BCEWithLogitsLossImpl pos_weight(Tensor setter);
