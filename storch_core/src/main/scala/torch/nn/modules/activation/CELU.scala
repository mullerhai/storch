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
import org.bytedeco.pytorch.{ELUOptions, SELUImpl, SELUOptions, SoftmaxOptions}
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{CELUImpl, CELUOptions}

/** Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the
  * elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
  *
  * Softmax is defined as: $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
  *
  * When the input Tensor is a sparse tensor then the unspecifed values are treated as ``-inf``.
  */
final class CELU[D <: DType: Default](
    alpha: Float = 1.0f,
    inplace: Boolean = false,
    size: Option[Int] = None
) extends TensorModule[D]:

  val options = if size.isDefined then CELUOptions(size.get) else CELUOptions()
  options.inplace().put(inplace)
  options.alpha().put(alpha.toDouble)

  override val nativeModule: CELUImpl = CELUImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  override def toString =
    getClass().getSimpleName() + "(size=" + size + ",alpha=" + alpha + ",inplace=" + inplace + ")"

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

object CELU {

  def apply[D <: DType: Default](
      alpha: Float = 1.0f,
      inplace: Boolean = false,
      size: Option[Int] = None
  ): CELU[D] =
    new CELU(alpha, inplace, size)
}
