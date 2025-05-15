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

import org.bytedeco.pytorch.{AdaptiveMaxPool1dImpl, AdaptiveMaxPool1dOptions, T_TensorTensor_T}
import org.bytedeco.pytorch

import torch.internal.NativeConverters.{fromNative, toNative, toOptional}
import org.bytedeco.pytorch.LongOptionalVector
import org.bytedeco.pytorch.LongOptional

/** Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  *
  * The output is of size H x W, for any input size. The number of output features is equal to the
  * number of input planes.
  */
final class AdaptiveMaxPool1d[D <: BFloat16 | Float32 | Float64: Default](
    outputSize: Int | Option[Int] | (Option[Int], Option[Int]) | (Int, Int),
    returnIndices: Boolean = false
) extends Module {
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private def nativeOutputSize = outputSize match
    case (h: Int, w: Int) =>
      toNative((h, w)) // new LongOptionalVector(new LongOptional(h), new LongOptional(w))
    case x: Int => toNative((x)) // new LongOptionalVector(new LongOptional(x), new LongOptional(x))
    // We know this can only be int so we can suppress the type test for Option[Int] cannot be checked at runtime warning
    case (h: Option[Int @unchecked], w: Option[Int @unchecked]) =>
      new LongOptionalVector(h.toOptional, w.toOptional)
    case x: Option[Int] =>
      new LongOptionalVector(x.toOptional, x.toOptional)

  private val options: AdaptiveMaxPool1dOptions = AdaptiveMaxPool1dOptions(nativeOutputSize)
  options.output_size().put(nativeOutputSize)
  override protected[torch] val nativeModule: AdaptiveMaxPool1dImpl = AdaptiveMaxPool1dImpl(options)

  override def hasBias(): Boolean = false

  def apply(t: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.forward(t.native)
  )
  def forward_with_indices(t: Tensor[D]): (Tensor[D], Tensor[D]) =
    val outputWithIndices = nativeModule.forward_with_indices(t.native)
    (fromNative(outputWithIndices.get0()), fromNative(outputWithIndices.get1()))
  override def toString =
    s"${getClass.getSimpleName}(outputSize=$outputSize returnIndices=$returnIndices)"
}

object AdaptiveMaxPool1d:
  def apply[DT <: BFloat16 | Float32 | Float64: Default](
      output_size: Int | Option[Int] | (Option[Int], Option[Int]) | (Int, Int),
      return_indices: Boolean = false
  ): AdaptiveMaxPool1d[DT] =
    new AdaptiveMaxPool1d(output_size, return_indices)
