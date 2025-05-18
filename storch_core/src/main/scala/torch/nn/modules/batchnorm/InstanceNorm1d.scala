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

package torch.nn.modules.batchnorm

import org.bytedeco.javacpp.{LongPointer, DoublePointer}
import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{InstanceNorm1dImpl, InstanceNormOptions}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.{HasWeight, TensorModule}
import torch.{Default, FloatNN, ComplexNN, Tensor}
import torch.internal.NativeConverters.{fromNative, toNative}
/** Applies Group Normalization over a mini-batch of inputs
  *
  * @param numGroups
  *   number of groups to separate the channels into
  * @param numChannels
  *   number of channels expected in input
  * @param eps
  *   a value added to the denominator for numerical stability
  * @param affine
  *   a boolean value that when set to `true`, this module has learnable per-channel affine
  *   parameters initialized to ones (for weights) and zeros (for biases)
  */
final class InstanceNorm1d[ParamType <: FloatNN | ComplexNN: Default](
    numFeatures: Int,
    eps: Float | Double = 1e-05f,
    momentum: Float | Option[Float] = 0.1f,
    affine: Boolean = false,
    trackRunningStats: Boolean = false
) extends HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: InstanceNormOptions = InstanceNormOptions(toNative(numFeatures)) 

  options.affine().put(affine)
  eps match {
    case e: Double => options.eps().put(e)
    case e: Float => options.eps().put(e.toDouble)
  }
  momentum match {
    case m: Float => options.momentum().put(DoublePointer(1).put(m.toDouble))
    case m: Option[Float] =>
      if m.isDefined then options.momentum().put(DoublePointer(1).put(m.get.toDouble))
  }

  options.num_features().put(LongPointer(1).put(numFeatures.toLong))
  options.track_running_stats.put(trackRunningStats)

  override private[torch] val nativeModule: InstanceNorm1dImpl = InstanceNorm1dImpl(options)

  val weight: Tensor[ParamType] = fromNative[ParamType](nativeModule.weight)
  val bias: Tensor[ParamType] = fromNative[ParamType](nativeModule.bias)

  override def hasBias(): Boolean = true

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def reset_running_stats(): Unit = nativeModule.reset_running_stats()

  def running_mean(): Tensor[ParamType] = fromNative(nativeModule.running_mean())

  def running_var(): Tensor[ParamType] = fromNative(nativeModule.running_var())

  def num_batches_tracked(): Tensor[ParamType] = fromNative(nativeModule.num_batches_tracked())

  def apply(t: Tensor[ParamType]): Tensor[ParamType] =
    fromNative[ParamType](nativeModule.forward(t.native))

  override def toString(): String =
    s"${getClass().getSimpleName()}(numFeatures=$numFeatures eps=$eps momentum=$momentum affine=$affine trackRunningStats=$trackRunningStats)"

object InstanceNorm1d:

  def apply[ParamType <: FloatNN | ComplexNN: Default](
      num_features: Int,
      eps: Float | Double = 1e-05f,
      momentum: Float | Option[Float] = 0.1f,
      affine: Boolean = false,
      track_running_stats: Boolean = false
  ): InstanceNorm1d[ParamType] =
    new InstanceNorm1d(num_features, eps, momentum, affine, track_running_stats)
