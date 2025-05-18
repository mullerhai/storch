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
package activation

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LeakyReLUImpl, LeakyReLUOptions}
import torch.internal.NativeConverters.fromNative

/** Applies the rectified linear unit function element-wise:
  *
  * $\text{ReLU}(x) = (x)^+ = \max(0, x)$
  */
final class LeakyReLU[D <: DType: Default](negativeSlope: Float, inplace: Boolean = false)
    extends TensorModule[D]:
  private val options = new LeakyReLUOptions()
  options.inplace().put(inplace)
  options.negative_slope().put(negativeSlope.toDouble)

  override protected[torch] val nativeModule: LeakyReLUImpl = LeakyReLUImpl(options)

  def reset(): Unit = nativeModule.reset()
  override def hasBias(): Boolean = false

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  override def toString =
    getClass().getSimpleName() + s"(negativeSlope=$negativeSlope,inplace=$inplace)"

object LeakyReLU:
  def apply[D <: DType: Default](
      negative_slope: Float = 0.01f,
      inplace: Boolean = false
  ): LeakyReLU[D] =
    new LeakyReLU[D](negative_slope, inplace)
