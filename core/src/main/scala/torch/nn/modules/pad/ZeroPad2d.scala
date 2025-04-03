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
package pad

import org.bytedeco.javacpp.{LongPointer, DoublePointer, IntPointer}

import org.bytedeco.pytorch.{LongOptional, LongOptionalVector, ZeroPad2dImpl, ZeroPad2dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.internal.NativeConverters.toOptional

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the
  * number of input planes.
  */
final class ZeroPad2d[D <: BFloat16 | Float32 | Float64: Default](
    padding: Int | (Int, Int, Int, Int)
) extends Module {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val paddingNative = padding match {
    case (top, bottom, left, right) => toNative(top, bottom, left, right)
    case x: Int =>
      LongPointer(Array(x.toLong, x.toLong, x.toLong, x.toLong)*) // IntPointer(Array(x,x,x,x)*)
    case _ => throw new IllegalArgumentException("padding must be a tuple of 2, 4 or 8 integers")
  }
  private val options: ZeroPad2dOptions = ZeroPad2dOptions(paddingNative)
  options.padding().put(paddingNative)

  override protected[torch] val nativeModule: ZeroPad2dImpl = ZeroPad2dImpl(
    options
  )

  override def hasBias(): Boolean = false

  override def toString =
    s"${getClass.getSimpleName}(padding = ${padding})"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.forward(t.native)
  )
}

object ZeroPad2d:
  def apply[D <: BFloat16 | Float32 | Float64: Default](
      padding: Int | (Int) | (Int, Int, Int, Int)
  ): ZeroPad2d[D] =
    new ZeroPad2d[D](padding)
