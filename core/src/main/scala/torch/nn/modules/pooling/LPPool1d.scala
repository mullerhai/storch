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
package pooling

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LPPool1dImpl, LPPool1dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final class LPPool1d[D <: BFloat16 | Float32 | Float64: Default](
    nornType: Float,
    kernelSize: Int | (Int, Int),
    stride: Int | (Int, Int),
    ceilMode: Boolean = false
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: LPPool1dOptions = LPPool1dOptions(toNative(kernelSize))

  stride match {
    case s: Int        => options.stride().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong)*)
  }
  kernelSize match {
    case s: Int        => options.kernel_size().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.kernel_size().put(Array(s._1.toLong, s._2.toLong)*)
  }

  options.ceil_mode().put(ceilMode)
  options.norm_type().put(DoublePointer(1).put(nornType.toDouble))
  override private[torch] val nativeModule: LPPool1dImpl = LPPool1dImpl(options)

  override def hasBias(): Boolean = false

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernelSize, stride=$stride, nornType=$nornType, ceilMode=$ceilMode)"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

object LPPool1d:
  def apply[D <: BFloat16 | Float32 | Float64: Default](
      norm_type: Float,
      kernel_size: Int | (Int),
      stride: Int | (Int),
      ceil_mode: Boolean = false
  ): LPPool1d[D] =
    new LPPool1d[D](norm_type, kernel_size, stride, ceil_mode)
