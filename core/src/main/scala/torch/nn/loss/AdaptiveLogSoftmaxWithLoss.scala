//package torch.nn.loss //AdaptiveLogSoftmaxWithLossImpl
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{
  AdaptiveLogSoftmaxWithLossImpl,
  AdaptiveLogSoftmaxWithLossOptions,
  LongVector
}

final class AdaptiveLogSoftmaxWithLoss(
    inFeatures: Long,
    nClasses: Long,
    cutoffs: Array[Long],
    divValue: Double,
    headBias: Boolean
) extends LossFunc {

  val cutoffsVec = LongVector(cutoffs*)
  private val options = new AdaptiveLogSoftmaxWithLossOptions(inFeatures, nClasses, cutoffsVec)
  options.div_value.put(divValue)
  options.head_bias.put(headBias)
//  options.nClasses = 1000

  override private[torch] val nativeModule: AdaptiveLogSoftmaxWithLossImpl =
    AdaptiveLogSoftmaxWithLossImpl(options)

  override def hasBias(): Boolean = false

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native).output()
  )

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native).output()
    )
  }
}

object AdaptiveLogSoftmaxWithLoss:
  def apply(
      in_features: Long,
      n_classes: Long,
      cutoffs: Array[Long],
      div_value: Double,
      head_bias: Boolean
  ): AdaptiveLogSoftmaxWithLoss =
    new AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value, head_bias)
