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

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{MaxUnpool2dImpl, MaxUnpool2dOptions, LongVectorOptional, LongVector}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final class MaxUnpool2d[D <: FloatNN | ComplexNN: Default](
    kernelSize: Int | (Int, Int),
    stride: Int | (Int, Int),
    padding: Int | (Int, Int) = 0
) extends TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: MaxUnpool2dOptions = kernelSize match {
    case k: Int        => MaxUnpool2dOptions(toNative((k, k)))
    case k: (Int, Int) => MaxUnpool2dOptions(toNative(k))
  }

  stride match {
    case s: Int        => options.stride().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong)*)
  }
  padding match {
    case s: Int        => options.padding().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.padding().put(Array(s._1.toLong, s._2.toLong)*)
  }
  kernelSize match {
    case s: Int        => options.kernel_size().put(Array(s.toLong, s.toLong)*)
    case s: (Int, Int) => options.kernel_size().put(Array(s._1.toLong, s._2.toLong)*)
  }

  override private[torch] val nativeModule: MaxUnpool2dImpl = MaxUnpool2dImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def toString(): String =
    s"${getClass.getSimpleName}(kernelSize=$kernelSize, stride=$stride, padding=$padding)"

  def apply(input: Tensor[D], indices: Tensor[Int64]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, indices.native)
  )

  def apply(input: Tensor[D], indices: Tensor[Int64], outputSize: Array[Int]): Tensor[D] =
    val out = outputSize.map(_.toLong)
    fromNative(
      nativeModule.forward(input.native, indices.native, LongVectorOptional(LongVector(out*)))
    )
  
  def forward(input: Tensor[D], indices: Tensor[Int64], outputSize: Array[Int]): Tensor[D] =
    val out = outputSize.map(_.toLong)
    fromNative(
      nativeModule.forward(input.native, indices.native, LongVectorOptional(LongVector(out*)))
    )
  
  def forward(input: Tensor[D], indices: Tensor[Int64]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, indices.native)
  )

  override def apply(v1: Tensor[D]): Tensor[D] = ???

object MaxUnpool2d:
  def apply[D <: FloatNN | ComplexNN: Default](
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int),
      padding: Int | (Int, Int) = 0
  ): MaxUnpool2d[D] =
    new MaxUnpool2d(kernel_size, stride, padding)

// outputSize: Array[Int]  =4
//  def apply(t: Tensor[D]): Tensor[D] = // fromNative(nativeModule.forward(t.native))
