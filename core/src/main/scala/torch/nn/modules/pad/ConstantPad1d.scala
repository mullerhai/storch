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

import org.bytedeco.pytorch.{
  LongOptional,
  LongOptionalVector,
  ConstantPad1dImpl,
  ConstantPad1dOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.internal.NativeConverters.toOptional

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the
  * number of input planes.
  */
final class ConstantPad1d[D <: BFloat16 | Float32 | Float64: Default](
    padding: Int | (Int, Int),
    value: Float | Double
) extends Module {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: ConstantPad1dOptions = ConstantPad1dOptions(toNative(padding))
  options.padding().put(toNative(padding))
//  options.value().put(DoublePointer(1).put(value.toDouble))
  value match {
    case v: Float => options.value().put(v.toDouble)
    case v: Double => options.value().put(v)
  }
  override protected[torch] val nativeModule: ConstantPad1dImpl = ConstantPad1dImpl(
    options
  )

  override def hasBias(): Boolean = false

  def apply(t: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.forward(t.native)
  )

  override def toString =
    s"${getClass.getSimpleName}(padding = ${padding}, value = ${value})"
}

object ConstantPad1d:
  def apply[D <: BFloat16 | Float32 | Float64: Default](
      padding: Int | (Int, Int),
      value: Float | Double
  ): ConstantPad1d[D] =
    new ConstantPad1d[D](padding, value)
