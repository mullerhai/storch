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
package linear

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LinearImpl, LinearOptions}
import internal.NativeConverters.fromNative

/** Applies a linear transformation to the incoming data: $y = xA^T + b$
  *
  * This module supports `TensorFloat32<tf32_on_ampere>`.
  *
  * Example:
  *
  * ```scala sc:nocompile
  * import torch.*
  *
  * val linear = nn.Linear[Float32](20, 30)
  * val input = torch.rand(Seq(128, 20))
  * println(linear(input).size) // ArraySeq(128, 30)
  * ```
  *
  * @group nn_linear
  * // dtype: ParamType = defaultDType[ParamType]
  * @param inFeatures
  *   size of each input sample
  * @param outFeatures
  *   size of each output sample
  * @param bias
  *   If set to ``false``, the layer will not learn an additive bias. Default: ``true``
  */
final class Linear[ParamType <: FloatNN | ComplexNN: Default](
    val in_features: Long,
    val out_features: Long,
    val add_bias: Boolean = true
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new LinearOptions(in_features, out_features)
  options.bias().put(add_bias)

  override private[torch] val nativeModule: LinearImpl = new LinearImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  override def hasBias(): Boolean = options.bias().get()

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def weight = fromNative[ParamType](nativeModule.weight())

  def weight_=(t: Tensor[ParamType]): Tensor[ParamType] =
    nativeModule.weight(t.native)
    t

  def bias = fromNative[ParamType](nativeModule.bias())
  def bias_=(t: Tensor[ParamType]): Tensor[ParamType] =
    nativeModule.bias(t.native)
    t

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )
  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )

  override def toString =
    s"${getClass.getSimpleName}(in_features=$in_features, out_features=$out_features, bias=$add_bias)"

object Linear:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      in_features: Long,
      out_features: Long,
      bias: Boolean = true
      // dtype: ParamType = defaultDType[ParamType]
  ): Linear[ParamType] = new Linear(in_features, out_features, bias)
