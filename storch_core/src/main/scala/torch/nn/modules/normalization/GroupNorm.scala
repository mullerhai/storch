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
import org.bytedeco.pytorch.{GroupNormImpl, GroupNormOptions}
import torch.internal.NativeConverters.fromNative

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
final class GroupNorm[ParamType <: FloatNN | ComplexNN: Default](
    numGroups: Int,
    numChannels: Int,
    eps: Double = 1e-05,
    affine: Boolean = true
) extends HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: GroupNormOptions = GroupNormOptions(numGroups, numChannels)
  options.eps().put(eps)
  options.affine().put(affine)

  override private[torch] val nativeModule: GroupNormImpl = GroupNormImpl(options)

  val weight: Tensor[ParamType] = fromNative[ParamType](nativeModule.weight)
  val bias: Tensor[ParamType] = fromNative[ParamType](nativeModule.bias)

  override def hasBias(): Boolean = true

  def reset(): Unit = nativeModule.reset()
  override def toString =
    s"${getClass.getSimpleName}(numGroups = ${numGroups}, numChannels = ${numChannels},eps=$eps affine=$affine)"

  def apply(t: Tensor[ParamType]): Tensor[ParamType] =
    fromNative[ParamType](nativeModule.forward(t.native))

object GroupNorm:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      num_groups: Int,
      num_channels: Int,
      eps: Double = 1e-05,
      affine: Boolean = true
  ): GroupNorm[ParamType] =
    new GroupNorm(num_groups, num_channels, eps, affine)
