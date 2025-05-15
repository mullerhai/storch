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

import org.bytedeco.javacpp.{LongPointer, DoublePointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{MaxUnpool3dOptions, PixelShuffleImpl, PixelShuffleOptions}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D max pooling over an input signal composed of several input planes. */
final class PixelShuffle[D <: BFloat16 | Float32 | Float64: Default](upscaleFactor: Int)
    extends TensorModule[D]:

//  private val options: PixelShuffleOptions = PixelShuffleOptions(toNative(upscaleFactor))
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options: PixelShuffleOptions = PixelShuffleOptions(LongPointer(upscaleFactor.toLong))
  options.upscale_factor.put(LongPointer(upscaleFactor))

  override private[torch] val nativeModule: PixelShuffleImpl = PixelShuffleImpl(options)

  override def hasBias(): Boolean = false

  override def toString(): String =
    s"${getClass.getSimpleName}(upscaleFactor=$upscaleFactor)"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

object PixelShuffle:
  def apply[D <: BFloat16 | Float32 | Float64: Default](upscale_factor: Int): PixelShuffle[D] =
    new PixelShuffle[D](upscale_factor)
