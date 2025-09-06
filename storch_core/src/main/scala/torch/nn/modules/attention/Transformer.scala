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
  TensorMapper,
  AnyModule,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorVector,
  TransformerImpl,
  TransformerOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.attention.Transformer.TransformerActivation

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv custom_encoder ,custom_decoder,
  */
final class Transformer[ParamType <: FloatNN | ComplexNN: Default](
    dModel: Int = 512,
    nhead: Int = 8,
    numEncoderLayers: Int = 6,
    numDecoderLayers: Int = 6,
    dimFeedforward: Int = 2048,
    dropout: Float = 0.1,
    activation: TransformerActivation | String = TransformerActivation.kReLU,
    customEncoder: Option[AnyModule] = None,
    customDecoder: Option[AnyModule] = None,
    layer_norm_eps: Float = 1e-05,
    batch_first: Boolean = false,
    norm_first: Boolean = false,
    bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new TransformerOptions(
    dModel.toLong,
    nhead.toLong,
    numEncoderLayers.toLong,
    numDecoderLayers.toLong
  )
  options.num_encoder_layers().put(LongPointer(1).put(numEncoderLayers.toLong))
  options.num_decoder_layers().put(LongPointer(1).put(numDecoderLayers.toLong))
  options.dim_feedforward().put(LongPointer(1).put(dimFeedforward.toLong))
  options.dropout().put(dropout.toDouble)
  options.nhead().put(LongPointer(1).put(nhead.toLong))
  options.d_model().put(LongPointer(1).put(dModel.toLong))

  if (customEncoder.isDefined) options.custom_encoder().put(customEncoder.get)
  if (customDecoder.isDefined) options.custom_decoder().put(customDecoder.get)

  activation match {
    case TransformerActivation.kReLU | "relu" | "Relu" | "RELU" | "ReLu" | "ReLU" =>
      options.activation().put(new kReLU)
    case TransformerActivation.kGELU | "gelu" | "Gelu" | "GELU" | "GeLu" | "GeLU" =>
      options.activation().put(new kGELU)

  }

  override private[torch] val nativeModule: TransformerImpl = TransformerImpl(options)
  nativeModule.to(paramType.toScalarType, false)
  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def generate_square_subsequent_mask(sz: Int): Tensor[ParamType] = fromNative(
    TransformerImpl.generate_square_subsequent_mask(sz.toLong)
  ) // generate_square_subsequent_mask

  def encoder(encoderModule: AnyModule) = nativeModule.encoder(encoderModule)

  def decoder(decoderModule: AnyModule) = nativeModule.decoder(decoderModule)

  def apply(
      src: Tensor[ParamType],
      tgt: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] = None,
      tgt_mask: Option[Tensor[ParamType]] = None,
      memory_mask: Option[Tensor[ParamType]] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] = None,
      memory_key_padding_mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    val fore =
      if (src_mask.isDefined)
        nativeModule.forward(
          src.native,
          tgt.native,
          src_mask.get.native,
          tgt_mask.get.native,
          memory_mask.get.native,
          src_key_padding_mask.get.native,
          tgt_key_padding_mask.get.native,
          memory_key_padding_mask.get.native
        )
      else nativeModule.forward(src.native, tgt.native)
    fromNative(fore)
  }
  def forward(
      src: Tensor[ParamType],
      tgt: Tensor[ParamType],
      src_mask: Option[Tensor[ParamType]] = None,
      tgt_mask: Option[Tensor[ParamType]] = None,
      memory_mask: Option[Tensor[ParamType]] = None,
      src_key_padding_mask: Option[Tensor[ParamType]] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] = None,
      memory_key_padding_mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    val fore =
      if (src_mask.isDefined)
        nativeModule.forward(
          src.native,
          tgt.native,
          src_mask.get.native,
          tgt_mask.get.native,
          memory_mask.get.native,
          src_key_padding_mask.get.native,
          tgt_key_padding_mask.get.native,
          memory_key_padding_mask.get.native
        )
      else nativeModule.forward(src.native, tgt.native)
    fromNative(fore)
  }


  override def hasBias(): Boolean = false

  override def toString =
    s"${getClass.getSimpleName}(dimFeedforward=$dimFeedforward, bias=$bias d_model=$dModel nhead=$nhead numEncoderLayers= ${numEncoderLayers} numDecoderLayers= ${numDecoderLayers} dropout=$dropout customEncoder=$customEncoder customDecoder=$customDecoder activation=$activation  )"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

// layer_norm_eps=1e-05, batch_first=False, norm_first=False,
object Transformer:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      d_model: Int = 512,
      nhead: Int = 8,
      num_encoder_layers: Int = 6,
      num_decoder_layers: Int = 6,
      dim_feedforward: Int = 2048,
      dropout: Float = 0.1,
      activation: TransformerActivation | String = TransformerActivation.kReLU,
      custom_encoder: Option[AnyModule] = None,
      custom_decoder: Option[AnyModule] = None,
      layer_norm_eps: Float = 1e-05,
      batch_first: Boolean = false,
      norm_first: Boolean = false,
      bias: Boolean = true
  ): Transformer[ParamType] = new Transformer[ParamType](
    d_model,
    nhead,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout,
    activation,
    custom_encoder,
    custom_decoder,
    layer_norm_eps,
    batch_first,
    norm_first,
    bias
  )
  enum TransformerActivation:
    case kReLU, kGELU, TensorMapper
