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

import org.bytedeco.javacpp.{LongPointer, DoublePointer}

import org.bytedeco.pytorch.{LongOptional, LongOptionalVector, ZeroPad1dImpl, ZeroPad1dOptions}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.internal.NativeConverters.toOptional

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the
  * number of input planes.
  */
final class ZeroPad1d[D <: BFloat16 | Float32 | Float64: Default](
    padding: Int | (Int, Int)
) extends Module {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: ZeroPad1dOptions = ZeroPad1dOptions(toNative(padding))
  options.padding().put(toNative(padding))

  override protected[torch] val nativeModule: ZeroPad1dImpl = ZeroPad1dImpl(
    options
  )

  override def hasBias(): Boolean = false

  override def toString =
    s"${getClass.getSimpleName}(padding = ${padding})"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.forward(t.native)
  )
}

object ZeroPad1d:
  def apply[D <: BFloat16 | Float32 | Float64: Default](
      padding: Int | (Int, Int)
  ): ZeroPad1d[D] =
    new ZeroPad1d[D](padding)
