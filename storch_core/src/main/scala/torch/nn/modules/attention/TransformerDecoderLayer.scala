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
package attention

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  kReLU,
  kGELU,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorVector,
  TransformerDecoderLayerImpl,
  TransformerDecoderLayerOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.attention.Transformer.TransformerActivation

import torch.nn.modules.attention.Transformer.TransformerActivation

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class TransformerDecoderLayer[ParamType <: FloatNN | ComplexNN: Default](
    dModel: Int,
    nHead: Int,
    dimFeedforward: Int = 2048,
    dropout: Float = 0.1,
    activation: TransformerActivation | String = TransformerActivation.kReLU,
    layerNormEps: Float = 1e-5,
    batchFirst: Boolean = false,
    normFirst: Boolean = false,
    bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  override def toString =
    s"${getClass.getSimpleName}(dModel=$dModel, nHead=$nHead activation=${activation.toString} dimFeedforward=$dimFeedforward dropout=  $dropout bias=$bias)"

  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val options = new TransformerDecoderLayerOptions(dModel.toLong, nHead.toLong)
  options.d_model().put(LongPointer(1).put(dModel.toLong))
  options.nhead().put(LongPointer(1).put(nHead.toLong))
  options.dim_feedforward().put(LongPointer(1).put(dimFeedforward.toLong))
  options.dropout().put(DoublePointer(1).put(dropout.toDouble))

  activation match {
    case TransformerActivation.kReLU | "relu" | "Relu" | "ReLU" | "RELU" | "ReLu" =>
      options.activation().put(new kReLU)
    case TransformerActivation.kGELU | "gelu" | "Gelu" | "GeLU" | "GELU" | "GeLu" =>
      options.activation().put(new kGELU)
  }

  override private[torch] val nativeModule: TransformerDecoderLayerImpl =
    TransformerDecoderLayerImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(
      tgt: Tensor[ParamType],
      memory: Tensor[ParamType],
      tgt_mask: Option[Tensor[ParamType]] = None,
      memory_mask: Option[Tensor[ParamType]] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] = None,
      memory_key_padding_mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    val fore =
      if (tgt_mask.isDefined)
        nativeModule.forward(
          tgt.native,
          memory.native,
          tgt_mask.get.native,
          memory_mask.get.native,
          tgt_key_padding_mask.get.native,
          memory_key_padding_mask.get.native
        )
      else nativeModule.forward(tgt.native, memory.native)
    fromNative(fore)
  }

  override def hasBias(): Boolean = false // options.bias().get()

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object TransformerDecoderLayer:

  def apply[PT <: FloatNN | ComplexNN: Default](
      d_model: Int,
      n_head: Int,
      dim_feedforward: Int = 2048,
      dropout: Float = 0.1,
      activation: TransformerActivation | String = TransformerActivation.kReLU,
      layer_norm_eps: Float = 1e-5,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      bias: Boolean = true
  ): TransformerDecoderLayer[PT] = new TransformerDecoderLayer[PT](
    d_model,
    n_head,
    dim_feedforward,
    dropout,
    activation,
    layer_norm_eps,
    batch_first,
    norm_first,
    bias
  )
