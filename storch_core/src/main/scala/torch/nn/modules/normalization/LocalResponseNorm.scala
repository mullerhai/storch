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
package normalization

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{BatchNormOptions, LocalResponseNormImpl, LocalResponseNormOptions}
import torch.internal.NativeConverters.fromNative

/** Applies Batch Normalization over a 2D or 3D input as described in the paper [Batch
  * Normalization: Accelerating Deep Network Training by Reducing Internal Covariate
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
  * @note
  *   This `momentum` argument is different from one used in optimizer classes and the conventional
  *   notion of momentum. Mathematically, the update rule for running statistics here is
  *   $\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t$,
  *   where $\hat{x}$ is the estimated statistic and $x_t$ is the new observed value.
  *
  * Because the Batch Normalization is done over the [C] dimension, computing statistics on [(N, L)]
  * slices, it\'s common terminology to call this Temporal Batch Normalization.
  *
  * Args:
  *
  * @param num_features
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
  *   - Input: $(N, C)$ or $(N, C, L)$, where $N$ is the batch size, $C$ is the number of features
  *     or channels, and $L$ is the sequence length
  *   - Output: $(N, C)$ or $(N, C, L)$ (same shape as input)
  *
  * @group nn_conv
  *
  * TODO use dtype
  */
final class LocalResponseNorm[ParamType <: FloatNN | ComplexNN: Default](
    size: Int,
    alpha: Double = 0.0001,
    beta: Double = 0.75,
    k: Double = 1.0
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new LocalResponseNormOptions(size)
  options.alpha().put(alpha)
  options.beta().put(beta)
  options.k().put(k)

  override private[torch] val nativeModule: LocalResponseNormImpl = LocalResponseNormImpl(options)
  nativeModule.to(paramType.toScalarType, false)


  override def hasBias(): Boolean = true

  def reset(): Unit = nativeModule.reset()
  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  override def toString(): String =
    s"${getClass().getSimpleName()}(size= ${size} beta= ${beta} k=${k} alpha=$alpha)"

  override def weight: Tensor[ParamType] = ???

object LocalResponseNorm:

  def apply[ParamType <: FloatNN | ComplexNN: Default](
      size: Int,
      alpha: Double = 0.0001,
      beta: Double = 0.75,
      k: Double = 1.0
  ): LocalResponseNorm[ParamType] = new LocalResponseNorm(size, alpha, beta, k)



