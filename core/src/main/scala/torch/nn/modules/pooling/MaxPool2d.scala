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
import org.bytedeco.pytorch.{MaxPool2dImpl, MaxPool2dOptions, T_TensorTensor_T}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final class MaxPool2d[D <: BFloat16 | Float32 | Float64 | Int64: Default](
    kernelSize: Int | (Int, Int),
    stride: Int | (Int, Int),
    padding: Int | (Int, Int) = 0,
    dilation: Int | (Int, Int) = 1,
    returnIndices: Boolean = false,
    ceilMode: Boolean = false
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: MaxPool2dOptions = kernelSize match {
    case k: Int        => MaxPool2dOptions(toNative((k, k)))
    case k: (Int, Int) => MaxPool2dOptions(toNative(k))
  }
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
  dilation match {
    case s: Int        => options.dilation().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.dilation().put(Array(s._1.toLong, s._2.toLong)*)
  }

  options.ceil_mode().put(ceilMode)
  override private[torch] val nativeModule: MaxPool2dImpl = MaxPool2dImpl(options)

  override def hasBias(): Boolean = false

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernelSize, stride=$stride, padding=$padding, dilation=$dilation, ceilMode=$ceilMode)"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward_with_indices(t: Tensor[D]): (Tensor[D], Tensor[D]) =
    val outputWithIndices = nativeModule.forward_with_indices(t.native)
    (fromNative(outputWithIndices.get0()), fromNative(outputWithIndices.get1()))
object MaxPool2d:
  def apply[D <: BFloat16 | Float32 | Float64 | Int64: Default](
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int),
      padding: Int | (Int, Int) = 0,
      dilation: Int | (Int, Int) = 1,
      return_indices: Boolean = false,
      ceil_mode: Boolean = false
  ): MaxPool2d[D] =
    new MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
