/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package nn
package functional

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  NLLLossOptions,
  MSELossOptions,
  KLDivLossReduction,
  L1LossOptions,
  KLDivLossOptions,
  kBatchMean,
  PoissonNLLLossOptions,
  kSum,
  kMean,
  kNone,
  LossReduction,
  BCEWithLogitsLossOptions,
  BCELossOptions,
  HingeEmbeddingLossOptions,
  MultiMarginLossOptions,
  CosineEmbeddingLossOptions,
  SmoothL1LossOptions,
  HuberLossOptions,
  SoftMarginLossOptions,
  MultiLabelMarginLossOptions,
  MultiLabelSoftMarginLossOptions,
  TripletMarginLossOptions,
  TripletMarginWithDistanceLossOptions,
  CTCLossOptions,
  MarginRankingLossOptions,
  CrossEntropyLossOptions
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative

// Loss functions
private[torch] trait Loss {

  /** Function that measures Binary Cross Entropy between target and input logits.
    *
    * TODO support weight, reduction, pos_weight
    *
    * @group nn_loss
    */
  def binaryCrossEntropyWithLogits[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.binary_cross_entropy_with_logits(
        input.native,
        target.native,
        BCEWithLogitsLossOptions()
      )
    )
  }

  // format: off
  // http://bytedeco.org/javacpp-presets/pytorch/apidocs/

  /** This criterion computes the cross entropy loss between input logits and target. See
   * [[torch.nn.loss.CrossEntropyLoss]] for details.
   *
   * **Shape:**
   *
   * * Input: Shape $(C)$, $(N,C)$ or $(N,C,d_1,d_2,...,d_K)$ with $K≥1$ in the case of K-dimensional
   * loss.
   * * Target: If containing class indices, shape $()$, $(N)$ or $(N,d_1,d_2,...,d_K)$ with $K≥1$ 
   * in the case of K-dimensional loss where each value should be between $[0,C)$. If containing class 
   * probabilities, same shape as the input and each value should be between [0,1][0,1].
   *
   * where:
   * * C = number of classes
   * * N = batch size​
   *
   * @example
   * ```scala
   * // Example of target with class indices
   * val input = torch.randn(3, 5, requires_grad=True)
   * val target = torch.randint(5, (3,), dtype=torch.int64)
   * val loss = F.cross_entropy(input, target)
   * loss.backward()
   *
   * // Example of target with class probabilities
   * val input = torch.randn(3, 5, requires_grad=True)
   * val target = torch.randn(3, 5).softmax(dim=1)
   * val loss = F.crossEntropy(input, target)
   * loss.backward()
   * ```
   *
   * @param input
   * Predicted unnormalized logits; see Shape section above for supported shapes.
   * @param target
   * Ground truth class indices or class probabilities; see Shape section below for supported
   * shapes.
   * @param weight
   * a manual rescaling weight given to each class. If given, has to be a Tensor of size C
   * @param size_average
   * Deprecated (see reduction). By default, the losses are averaged over each loss element in
   * the batch. Note that for some losses, there multiple elements per sample. If the field
   * `size_average` is set to `false`, the losses are instead summed for each mini-batch. Ignored
   * when reduce is `false`. Default: `true`
   * @param ignore_index
   * Specifies a target value that is ignored and does not contribute to the input gradient. When
   * `size_average` is `true`, the loss is averaged over non-ignored targets. Note that
   * `ignore_index` is only applicable when the target contains class indices. Default: `-100`
   * @param reduce
   * Deprecated (see reduction). By default, the losses are averaged or summed over observations
   * for each mini-batch depending on `size_average`. When reduce is `false`, returns a loss per
   * batch element instead and ignores size_average. Default: `true`
   * @param reduction
   * Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
   * reduction will be applied, 'mean': the sum of the output will be divided by the number of
   * elements in the output, 'sum': the output will be summed. Note: `size_average` and `reduce`
   * are in the process of being deprecated, and in the meantime, specifying either of those two
   * args will override reduction. Default: 'mean'
   * @param label_smoothing
   * A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0
   * means no smoothing. The targets become a mixture of the original ground truth and a uniform
   * distribution as described in
   * [[https://arxiv.org/abs/1512.00567 Rethinking the Inception Architecture for Computer Vision]].
   * Default: 0.0
   * @return
   * [[torch.Tensor]]
   * @see
   * See
   * [[https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html torch.nn.functional.cross_entropy]]
   * @see
   * See
   * [[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for equivalent torch.nn.CrossEntropyLoss class]]
   * @see
   * See [[https://pytorch.org/cppdocs/ PyTorch C++ documentation]]
   * @see
   * See [[http://bytedeco.org/javacpp-presets/pytorch/apidocs/ ByteDeco PyTorch preset]]
   */
  // format: on

  def crossEntropy[
      I <: BFloat16 | Float32 | Float64,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[I] =
    fromNative(
      torchNative.cross_entropy(
        input.native,
        target.native
      )
    )

  def crossEntropy[
      I <: BFloat16 | Float32 | Float64,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[I],
      labelSmoothing: Double,
      ignoreIndex: Long,
      reduction: String
  ): Tensor[I] = {
    val options = CrossEntropyLossOptions()
    options.label_smoothing().put(labelSmoothing)
    options.ignore_index().put(ignoreIndex)
    options.weight().put(weight.native)
    val reductionNative = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }

    options.reduction().put(LossReduction(reductionNative))

    fromNative(
      torchNative.cross_entropy(
        input.native,
        target.native,
        options
      )
    )

  }
  def binary_cross_entropy_with_logits[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[O],
      pos_weight: Tensor[O],
      sizeAverage: Boolean,
      ignoreIndex: Long,
      reduce: Boolean,
      reduction: String,
      labelSmoothing: Float
  ): Tensor[O] = {
    val options = BCEWithLogitsLossOptions()
    options.weight().put(weight.native)
    options.pos_weight().put(pos_weight.native)
    val reductionNative = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(reductionNative))

    fromNative(torchNative.binary_cross_entropy_with_logits(input.native, target.native, options))
  }

  def binaryCrossEntropy[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[O],
      sizeAverage: Boolean,
      ignoreIndex: Long,
      reduce: Boolean,
      reduction: String,
      labelSmoothing: Float
  ): Tensor[O] = {

    val options = BCELossOptions()
    options.weight().put(weight.native)
    val reductionNative = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(reductionNative))
    fromNative(torchNative.binary_cross_entropy(input.native, target.native, options))
  }

  def poissonNllLoss[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input: Tensor[I],
      target: Tensor[O],
      logInput: Boolean,
      full: Boolean,
      eps: Float,
      reduction: String = "mean"
  ): Tensor[O] = {
    val options = PoissonNLLLossOptions()
    options.log_input().put(logInput)
    options.full().put(full)
    options.eps().put(eps)
    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))
    fromNative(
      torchNative.poisson_nll_loss(
        input.native,
        target.native,
        options
      )
    )

  }

  def consineEmbeddingLoss[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input1: Tensor[I],
      input2: Tensor[I],
      input3: Tensor[I],
      margin: Float,
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[O] = {
    val options = CosineEmbeddingLossOptions()
    options.margin().put(margin)

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.cosine_embedding_loss(
        input1.native,
        input2.native,
        input3.native,
        options
      )
    )

  }

  def crossEntropy[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[O],
      sizeAverage: Boolean,
      ignoreIndex: Long,
      reduce: Boolean,
      reduction: String,
      labelSmoothing: Float
  ): Tensor[O] = {
    val options = CrossEntropyLossOptions()
    options.weight().put(weight.native)
    options.ignore_index().put(ignoreIndex)
    options.label_smoothing().put(labelSmoothing)
    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(torchNative.cross_entropy(input.native, target.native, options))

  }

  //  def ctcLoss[]

  //  def gaussianNllLoss[D <:DType](
  //      input: Tensor[D],
  //      target: Tensor[D],
  //      p: Double,
  //      sizeAverage: Boolean,
  //      reduce: Boolean,
  //      reduction: String
  //  ): Tensor[D] =
  //
  //    fromNative(
  //    torchNative.gaussian_nll_loss(
  //      input.native,
  //      target.native,
  //      p,
  //      sizeAverage,
  //      reduce,
  //      reduction
  //    )
  //  )

  def hingeEmbeddingLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      margin: Double,
      p: Double,
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {

    val options = HingeEmbeddingLossOptions()
    options.margin().put(margin)
    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))
    fromNative(
      torchNative.hinge_embedding_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def klDiv[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      sizeAverage: Boolean,
      reduce: Boolean,
      log_target: Boolean,
      reduction: String
  ): Tensor[D] = {

    val options = KLDivLossOptions()
    options.log_target().put(log_target)

    val nativeReduction = reduction match {
      case "mean" | "Mean"                         => new kMean
      case "sum" | "Sum"                           => new kSum
      case "none" | "None"                         => new kNone
      case "batchMean" | "BatchMean" | "batchmean" => new kBatchMean
    }
    options.reduction().put(KLDivLossReduction(nativeReduction))

    fromNative(
      torchNative.kl_div(
        input.native,
        target.native,
        options
      )
    )

  }

  def l1Loss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = L1LossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.l1_loss(
        input.native,
        target.native,
        options
      )
    )

  }

  def mseLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = MSELossOptions()
    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.mse_loss(
        input.native,
        target.native,
        options
      )
    )

  }

  def marginRankingLoss[D <: DType](
      input1: Tensor[D],
      input2: Tensor[D],
      target: Tensor[D],
      margin: Double,
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = MarginRankingLossOptions()
    options.margin().put(margin)

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.margin_ranking_loss(
        input1.native,
        input2.native,
        target.native,
        options
      )
    )

  }

  def multilabelMarginLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      margin: Double,
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = MultiLabelMarginLossOptions()
    //    options.margin().put(margin)

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.multilabel_margin_loss(
        input.native,
        target.native,
        options
      )
    )

  }

  def multilabelSoftMarginLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      weight: Tensor[D],
      margin: Double,
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = MultiLabelSoftMarginLossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))
    options.weight().put(weight.native)
    fromNative(
      torchNative.multilabel_soft_margin_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def multiMarginLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      p: Double,
      margin: Double,
      weight: Tensor[D],
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = MultiMarginLossOptions()
    options.p().put(p.toLong)
    options.margin().put(margin)
    options.weight().put(weight.native)
    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.multi_margin_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def nllLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      weight: Tensor[D],
      sizeAverage: Boolean,
      ignoreIndex: Int,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = NLLLossOptions()
    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }

    options.reduction().put(LossReduction(nativeReduction))
    options.weight().put(weight.native)
    options.ignore_index().put(ignoreIndex.toLong)
    fromNative(
      torchNative.nll_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def nllLoss2d[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      weight: Tensor[D],
      sizeAverage: Boolean,
      ignoreIndex: Int,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    //    val options =
    //  val nativeReduction = reduction match {
    //    case "mean" | "Mean" => new kMean
    //    case "sum" | "Sum" => new kSum
    //    case "none" | "None" => new kNone
    //  }
    //  options.reduction().put(nativeReduction)

    fromNative(
      torchNative.nll_loss2d(
        input.native,
        target.native
      )
    )
  }

  def poissonNLLLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      logInput: Boolean,
      full: Boolean,
      sizeAverage: Boolean,
      eps: Double,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = PoissonNLLLossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))
    options.eps().put(eps)
    options.full().put(full)
    options.log_input().put(logInput)
    fromNative(
      torchNative.poisson_nll_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def huberLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      delta: Double,
      reduction: String
  ): Tensor[D] = {
    val options = HuberLossOptions()
    options.delta().put(delta)

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.huber_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def smoothL1Loss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      beta: Double,
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = SmoothL1LossOptions()
    options.beta().put(beta)

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.smooth_l1_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def softMarginLoss[D <: DType](
      input: Tensor[D],
      target: Tensor[D],
      reduction: String
  ): Tensor[D] = {
    val options = SoftMarginLossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.soft_margin_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def tripletMarginLoss[D <: DType](
      input1: Tensor[D],
      input2: Tensor[D],
      input3: Tensor[D],
      margin: Double,
      p: Double,
      eps: Double,
      swap: Boolean,
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = TripletMarginLossOptions()
    options.p().put(p)
    options.eps().put(eps)
    options.swap().put(swap)
    options.margin().put(margin)

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.triplet_margin_loss(input1.native, input2.native, input3.native, options)
    )
  }

  def tripletMarginWithDistanceLoss[D <: DType](
      input1: Tensor[D],
      input2: Tensor[D],
      input3: Tensor[D],
      margin: Double,
      p: Double,
      eps: Double,
      swap: Boolean,
      distanceFunction: String,
      sizeAverage: Boolean,
      reduce: Boolean,
      reduction: String
  ): Tensor[D] = {
    val options = TripletMarginWithDistanceLossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" => new kMean
      case "sum" | "Sum"   => new kSum
      case "none" | "None" => new kNone
    }
    options.reduction().put(LossReduction(nativeReduction))
    options.swap().put(swap)
    options.margin().put(margin)
//      options.distance_function().put(Pointer(distanceFunction))

    fromNative(
      torchNative.triplet_margin_with_distance_loss(
        input1.native,
        input2.native,
        input3.native,
        options
      )
    )
  }
}
