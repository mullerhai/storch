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

import org.bytedeco.pytorch.global.torch as torchNative

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  kReLU,
  kGELU,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorVector,
  TransformerEncoderLayerImpl,
  TransformerEncoderLayerOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.attention.Transformer.TransformerActivation

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class TransformerEncoderLayer[ParamType <: FloatNN | ComplexNN: Default](
    dModel: Int,
    nHead: Int,
    dimFeedforward: Int = 2048,
    dropout: Float | Double = 0.1,
    activation: TransformerActivation | String = TransformerActivation.kReLU,
    layerNormEps: Float = 1e-5,
    batchFirst: Boolean = false,
    normFirst: Boolean = false,
    bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  override def toString =
    s"${getClass.getSimpleName}(dModel=$dModel, nHead=$nHead activation=${activation.toString} dimFeedforward=$dimFeedforward dropout= $dropout bias=$bias)"

  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val options = new TransformerEncoderLayerOptions(dModel.toLong, nHead.toLong)
  options.d_model().put(dModel)
  options.nhead().put(nHead.toLong)
  options.dim_feedforward().put(LongPointer(1).put(dimFeedforward.toLong))
  dropout match {
    case d: Double => options.dropout().put(DoublePointer(1).put(d))
    case d: Float  => options.dropout().put(DoublePointer(1).put(d.toDouble))
  }

  activation match {
    case TransformerActivation.kReLU | "relu" | "Relu" | "ReLU" | "RELU" | "ReLu" =>
      options.activation().put(new kReLU)
    case TransformerActivation.kGELU | "gelu" | "Gelu" | "GeLU" | "GELU" | "GeLu" =>
      options.activation().put(new kGELU)
  }

  override private[torch] val nativeModule: TransformerEncoderLayerImpl =
    TransformerEncoderLayerImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(
      src: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None
  ): Tensor[ParamType] = {
    this.forward(src, src_mask, src_key_padding_mask)
  }
  def forward(
      src: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] | Tensor[ParamType] = None
  ): Tensor[ParamType] = {
    val srcMask = src_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
    }

    val srcKPM = src_key_padding_mask match {
      case k: Tensor[ParamType]         => k.native
      case k: Option[Tensor[ParamType]] => if k.isDefined then k.get.native else torchNative.empty()
    }
    val fore =
      if (srcMask.equals(torchNative.empty()) && srcKPM.equals(torchNative.empty()))
        nativeModule.forward(src.native)
      else nativeModule.forward(src.native, srcMask, srcKPM)
    fromNative(fore)
  }

  override def hasBias(): Boolean = false // options.bias().get()

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def apply(src: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(src.native)
  )

object TransformerEncoderLayer:
  def apply[PT <: FloatNN | ComplexNN: Default](
      d_model: Int,
      n_head: Int,
      dim_feedforward: Int = 2048,
      dropout: Float | Double = 0.1,
      activation: TransformerActivation | String = TransformerActivation.kReLU,
      layer_norm_eps: Float = 1e-5,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      bias: Boolean = true
  ): TransformerEncoderLayer[PT] = new TransformerEncoderLayer[PT](
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

//  options.d_model().put(LongPointer(1).put(dModel.toLong))
//  options.nhead().put(LongPointer(1).put(nHead.toLong))
