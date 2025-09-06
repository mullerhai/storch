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
  val options = new AdaptiveLogSoftmaxWithLossOptions(inFeatures, nClasses, cutoffsVec)
  options.div_value.put(divValue)
  options.head_bias.put(headBias)
//  options.nClasses = 1000

  override private[torch] val nativeModule: AdaptiveLogSoftmaxWithLossImpl =
    AdaptiveLogSoftmaxWithLossImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def head_size(): Long = nativeModule.head_size()

  def n_clusters(): Long = nativeModule.n_clusters()

  def shortlist_size(): Long = nativeModule.shortlist_size()

  def predict[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.predict(input.native)
  )

  def cutoffs_() = nativeModule.cutoffs() // todo make as Seq[Long]

  def log_prob[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.log_prob(input.native)
  )

  def get_full_log_prob[D <: DType](input: Tensor[D], head_output: Tensor[D]): Tensor[D] =
    fromNative(nativeModule._get_full_log_prob(input.native, head_output.native))

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native).output()
  )
  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native).output()
    )
  }
}
//  /** Given input tensor, and output of {@code head}, computes the log of the full
//   *  distribution */
//  public native @ByVal Tensor _get_full_log_prob(@Const @ByRef Tensor input, @Const @ByRef Tensor head_output);
//
//  /** Computes log probabilities for all n_classes */
//  public native @ByVal Tensor log_prob(@Const @ByRef Tensor input);
//
//  /** This is equivalent to {@code log_pob(input).argmax(1)} but is more efficient in
//   *  some cases */
//  public native @ByVal Tensor predict(@Const @ByRef Tensor input);
//
//  /** The options with which this {@code Module} was constructed */
//  public native @ByRef AdaptiveLogSoftmaxWithLossOptions options(); public native AdaptiveLogSoftmaxWithLossImpl options(AdaptiveLogSoftmaxWithLossOptions setter);
//
//  /** Cutoffs used to assign targets to their buckets. It should be an ordered
//   *  Sequence of integers sorted in the increasing order */
//  public native @ByRef @Cast("std::vector<int64_t>*") LongVector cutoffs(); public native AdaptiveLogSoftmaxWithLossImpl cutoffs(LongVector setter);
//
//  public native @Cast("int64_t") long shortlist_size(); public native AdaptiveLogSoftmaxWithLossImpl shortlist_size(long setter);
//
//  /** Number of clusters */
//  public native @Cast("int64_t") long n_clusters(); public native AdaptiveLogSoftmaxWithLossImpl n_clusters(long setter);
//
//  /** Output size of head classifier */
//  public native @Cast("int64_t") long head_size(); public native AdaptiveLogSoftmaxWithLossImpl head_size(long setter);
object AdaptiveLogSoftmaxWithLoss:
  def apply(
      in_features: Long,
      n_classes: Long,
      cutoffs: Array[Long],
      div_value: Double,
      head_bias: Boolean
  ): AdaptiveLogSoftmaxWithLoss =
    new AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value, head_bias)
