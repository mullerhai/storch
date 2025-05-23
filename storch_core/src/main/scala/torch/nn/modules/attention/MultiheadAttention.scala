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
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  MultiheadAttentionImpl,
  MultiheadAttentionOptions
}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class MultiheadAttention[ParamType <: FloatNN | ComplexNN: Default](
    embedDim: Int,
    numHeads: Int,
    dropout: Float = 0.0f,
    bias: Boolean = true,
    addBiasKV: Boolean = false,
    addZeroAttn: Boolean = false,
    kDim: Int | Option[Int] = None,
    vDim: Int | Option[Int] = None,
    batchFirst: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new MultiheadAttentionOptions(embedDim.toLong, numHeads.toLong)
  options.embed_dim().put(LongPointer(1).put(embedDim.toLong))
  options.num_heads().put(LongPointer(1).put(numHeads.toLong))
  options.dropout().put(DoublePointer(1).put(dropout.toDouble))
  options.bias().put(bias)
  options.add_bias_kv().put(addBiasKV)
  options.add_zero_attn().put(addZeroAttn)

  kDim match {
    case k: Int => options.kdim().put(k.toLong)
    case k: Option[Int] =>
      if k.isDefined then options.kdim().put(LongPointer(1).put(k.get.toLong))
      else options.kdim().put(LongPointer(1).put(embedDim.toLong))
  }
  vDim match {
    case v: Int => options.vdim().put(v.toLong)
    case v: Option[Int] =>
      if v.isDefined then options.vdim().put(LongPointer(1).put(v.get.toLong))
      else options.vdim().put(LongPointer(1).put(embedDim.toLong))
  }

  override private[torch] val nativeModule: MultiheadAttentionImpl = MultiheadAttentionImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(
      query: Tensor[ParamType],
      key: Tensor[ParamType],
      value: Tensor[ParamType]
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore = nativeModule.forward(query.native, key.native, value.native)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def apply(
      query: Tensor[ParamType],
      key: Tensor[ParamType],
      value: Tensor[ParamType],
      key_padding_mask: Tensor[ParamType],
      need_weights: Boolean,
      attn_mask: Tensor[ParamType],
      average_attn_weights: Boolean
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore = nativeModule.forward(
      query.native,
      key.native,
      value.native,
      key_padding_mask.native,
      need_weights,
      attn_mask.native,
      average_attn_weights
    )
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  def qkv_same_embed_dim(): Boolean = nativeModule._qkv_same_embed_dim()

  def in_proj_weight(weight: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.in_proj_weight(weight.native)

  def in_proj_bias(bias: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.in_proj_bias(bias.native)

  def bias_k(bias: Tensor[ParamType]): MultiheadAttentionImpl = nativeModule.bias_k(bias.native)

  def bias_v(bias: Tensor[ParamType]): MultiheadAttentionImpl = nativeModule.bias_v(bias.native)

  def q_proj_weight(weight: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.q_proj_weight(weight.native)

  def k_proj_weight(weight: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.k_proj_weight(weight.native)

  def v_proj_weight(weight: Tensor[ParamType]): MultiheadAttentionImpl =
    nativeModule.v_proj_weight(weight.native)

  def in_proj_weight(): Tensor[ParamType] = fromNative(nativeModule.in_proj_weight())

  def in_proj_bias(): Tensor[ParamType] = fromNative(nativeModule.in_proj_bias())

  def bias_k(): Tensor[ParamType] = fromNative(nativeModule.bias_k())

  def bias_v(): Tensor[ParamType] = fromNative(nativeModule.bias_v())

  def q_proj_weight(): Tensor[ParamType] = fromNative(nativeModule.q_proj_weight())

  def k_proj_weight(): Tensor[ParamType] = fromNative(nativeModule.k_proj_weight())

  def v_proj_weight(): Tensor[ParamType] = fromNative(nativeModule.v_proj_weight())

  def head_dim(): Long = nativeModule.head_dim()

  override def hasBias(): Boolean = options.bias().get()

  override def toString(): String =
    s"${getClass().getSimpleName()}(embedDim=$embedDim numHeads=$numHeads dropout=$dropout bias=$bias addBiasKV=$addBiasKV addZeroAttn=$addZeroAttn kDim=$kDim vDim=$vDim)"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object MultiheadAttention:

  def apply[ParamType <: FloatNN | ComplexNN: Default](
      embed_dim: Int,
      num_heads: Int,
      dropout: Float = 0.0f,
      bias: Boolean = true,
      add_bias_kv: Boolean = false,
      add_zero_attn: Boolean = false,
      kdim: Int | Option[Int] = None,
      vdim: Int | Option[Int] = None,
      batch_first: Boolean = false
  ): MultiheadAttention[ParamType] = new MultiheadAttention[ParamType](
    embed_dim,
    num_heads,
    dropout,
    bias,
    add_bias_kv,
    add_zero_attn,
    kdim,
    vdim,
    batch_first
  )
  enum TransformerActivation:
    case kReLU, kGELU, TensorMapper

//  options.kdim().put(LongPointer(1).put(kDim.toLong))
//  options.vdim().put(LongPointer(1).put(vDim.toLong))
//  public native @Cast("bool") boolean _qkv_same_embed_dim(); public native MultiheadAttentionImpl _qkv_same_embed_dim(boolean setter);
//  public native @ByRef Tensor in_proj_weight(); public native MultiheadAttentionImpl in_proj_weight(Tensor setter);
//  public native @ByRef Tensor in_proj_bias(); public native MultiheadAttentionImpl in_proj_bias(Tensor setter);
//  public native @ByRef Tensor bias_k(); public native MultiheadAttentionImpl bias_k(Tensor setter);
//  public native @ByRef Tensor bias_v(); public native MultiheadAttentionImpl bias_v(Tensor setter);
//  public native @ByRef Tensor q_proj_weight(); public native MultiheadAttentionImpl q_proj_weight(Tensor setter);
//  public native @ByRef Tensor k_proj_weight(); public native MultiheadAttentionImpl k_proj_weight(Tensor setter);
//  public native @ByRef Tensor v_proj_weight(); public native MultiheadAttentionImpl v_proj_weight(Tensor setter);
//  public native @Cast("int64_t") long head_dim(); public native MultiheadAttentionImpl head_dim(long setter);
