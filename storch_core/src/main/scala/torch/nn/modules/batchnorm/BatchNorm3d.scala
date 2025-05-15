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
package modules
package batchnorm

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch.{BatchNorm3dImpl, BatchNormOptions, DoubleOptional}
import org.bytedeco.pytorch
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies Batch Normalization over a 4D input as described in the paper [Batch Normalization:
  * Accelerating Deep Network Training by Reducing Internal Covariate
  * Shift](https://arxiv.org/abs/1502.03167) .
  *
  * $$y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$
  *
  * The mean and standard-deviation are calculated per-dimension over the mini-batches and $\gamma$
  * and $\beta$ are learnable parameter vectors of size [C] (where [C] is the number of features or
  * channels of the input). By default, the elements of $\gamma$ are set to 1 and the elements of
  * $\beta$ are set to 0. The standard-deviation is calculated via the biased estimator, equivalent
  * to *[torch.var(input, unbiased=False)]*.
  *
  * Also by default, during training this layer keeps running estimates of its computed mean and
  * variance, which are then used for normalization during evaluation. The running estimates are
  * kept with a default `momentum` of 0.1.
  *
  * If `trackRunningStats` is set to `false`, this layer then does not keep running estimates, and
  * batch statistics are instead used during evaluation time as well.
  *
  * Example:
  *
  * ```scala sc
  * import torch.nn
  * // With Learnable Parameters
  * var m = nn.BatchNorm2d(num_features = 100)
  * // Without Learnable Parameters
  * m = nn.BatchNorm2d(100, affine = false)
  * val input = torch.randn(Seq(20, 100, 35, 45))
  * val output = m(input)
  * ```
  *
  * @note
  *   This `momentum` argument is different from one used in optimizer classes and the conventional
  *   notion of momentum. Mathematically, the update rule for running statistics here is
  *   $\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t$,
  *   where $\hat{x}$ is the estimated statistic and $x_t$ is the new observed value.
  *
  * Because the Batch Normalization is done over the C dimension, computing statistics on (N, H, W)
  * slices, it’s common terminology to call this Spatial Batch Normalization.
  *
  * @param numFeatures
  *   number of features or channels $C$ of the input
  * @param eps:
  *   a value added to the denominator for numerical stability. Default: 1e-5
  * @param momentum
  *   the value used for the runningVean and runningVar computation. Can be set to `None` for
  *   cumulative moving average (i.e. simple average). Default: 0.1
  * @param affine:
  *   a boolean value that when set to `true`, this module has learnable affine parameters. Default:
  *   `True`
  * @param trackRunningStats:
  *   a boolean value that when set to `true`, this module tracks the running mean and variance, and
  *   when set to `false`, this module does not track such statistics, and initializes statistics
  *   buffers `runningMean` and `runningVar` as `None`. When these buffers are `None`, this module
  *   always uses batch statistics. in both training and eval modes. Default: `true`
  *
  * Shape:
  *
  *   - Input: $(N, C, H, W)$
  *   - Output: $(N, C, H, W)$ (same shape as input)
  *
  * @group nn_conv
  *
  * TODO use dtype
  */
final class BatchNorm3d[ParamType <: FloatNN | ComplexNN: Default](
    numFeatures: Int,
    eps: Float | Double = 1e-05f,
    momentum: Float | Option[Float] = 0.1f,
    affine: Boolean = true,
    trackRunningStats: Boolean = true
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new BatchNormOptions(toNative(numFeatures))   //LongPointer(1).put(numFeatures.toLong))
//  options.eps().put(DoublePointer(1).put(eps.toDouble))
  eps match {
    case e: Double => options.eps().put(e)
    case e: Float => options.eps().put(e.toDouble)
  }
  momentum match {
    case m: Float => options.momentum().put(DoublePointer(1).put(m.toDouble))
    case m: Option[Float] =>
      if m.isDefined then options.momentum().put(DoublePointer(1).put(m.get.toDouble))
  }

  options.affine().put(affine)
  options.track_running_stats().put(trackRunningStats)

  override private[torch] val nativeModule: BatchNorm3dImpl = BatchNorm3dImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  // TODO weight, bias etc. are undefined if affine = false. We need to take that into account
  val weight: Tensor[ParamType] = fromNative[ParamType](nativeModule.weight)
  val bias: Tensor[ParamType] = fromNative[ParamType](nativeModule.bias)
  // TODO running_mean, running_var, num_batches_tracked

  override def hasBias(): Boolean = true

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  override def toString(): String =
    s"${getClass().getSimpleName()}(numFeatures=$numFeatures eps=$eps momentum=$momentum affine=$affine trackRunningStats=$trackRunningStats)"

object BatchNorm3d:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      num_features: Int,
      eps: Float | Double = 1e-05f,
      momentum: Float | Option[Float] = 0.1f,
      affine: Boolean = true,
      track_running_stats: Boolean = true
  ): BatchNorm3d[ParamType] =
    new BatchNorm3d[ParamType](num_features, eps, momentum, affine, track_running_stats)
