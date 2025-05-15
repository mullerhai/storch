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

import org.bytedeco.pytorch.{AvgPool2dImpl, AvgPool2dOptions}
import org.bytedeco.pytorch

import torch.internal.NativeConverters.{fromNative, toNative, toOptional}
import org.bytedeco.pytorch.LongOptionalVector
import org.bytedeco.pytorch.LongOptional
import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the
  * number of input planes.
  */
final class AvgPool2d[D <: BFloat16 | Float32 | Float64: Default](
    kernelSize: Int | (Int, Int),
    stride: Int | (Int, Int),
    padding: Int | (Int, Int) = 0,
    ceilMode: Boolean = false,
    countIncludePad: Boolean = true,
    divisorOverride: Option[Int] = None
) extends Module {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val options = new AvgPool2dOptions(toNative(kernelSize)) 
//  kernelSize match {
//    case k: Int        => new AvgPool2dOptions(toNative((k, k)))
//    case k: (Int, Int) => new AvgPool2dOptions(toNative(k))
//  }

  stride match {
    case s: Int        => options.stride().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong)*)
  }
  kernelSize match {
    case s: Int        => options.kernel_size().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.kernel_size().put(Array(s._1.toLong, s._2.toLong)*)
  }
  padding match {
    case s: Int        => options.padding().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.padding().put(Array(s._1.toLong, s._2.toLong)*)
  }

  options.ceil_mode().put(ceilMode)
  options.count_include_pad().put(countIncludePad)
  if divisorOverride.isDefined then options.divisor_override().put(toOptional(divisorOverride))

  override protected[torch] val nativeModule: AvgPool2dImpl = AvgPool2dImpl(
    options
  )

  override def hasBias(): Boolean = false

  def apply(t: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.forward(t.native)
  )

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernelSize, stride=$stride, padding=$padding, countIncludePad=$countIncludePad, divisorOverride=${divisorOverride} ceilMode=$ceilMode)"

}

object AvgPool2d:
  def apply[DT <: BFloat16 | Float32 | Float64: Default](
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int),
      padding: Int | (Int, Int) = 0,
      ceil_mode: Boolean = false,
      count_include_pad: Boolean = true,
      divisor_override: Option[Int] = None
  ): AvgPool2d[DT] =
    new AvgPool2d(
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override
    )
