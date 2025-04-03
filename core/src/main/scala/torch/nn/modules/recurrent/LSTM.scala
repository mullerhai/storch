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
package recurrent

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  LSTMImpl,PackedSequence,
  LSTMOptions,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorVector,
  kCircular,
  kReflect,
  kReplicate,
  kZeros
}
import torch.internal.NativeConverters.{fromNative, toNative}

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class LSTM[ParamType <: FloatNN | ComplexNN: Default](
    inputSize: Int,
    hiddenSize: Int,
    numLayers: Int,
    bias: Boolean = true,
    batchFirst: Boolean = false,
    dropout: Float = 0.1f,
    bidirectional: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  type PackedSequenceTensorTensor = (PackedSequence, Tensor[ParamType], Tensor[ParamType]) //T_PackedSequenceT_TensorTensor_T_T
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new LSTMOptions(inputSize.toLong, hiddenSize.toLong)
  options.input_size().put(LongPointer(1).put(inputSize.toLong))
  options.hidden_size().put(LongPointer(1).put(hiddenSize.toLong))
  options.num_layers().put(LongPointer(1).put(numLayers.toLong))

  options.bias().put(bias)
  options.batch_first().put(batchFirst)
  options.dropout().put(dropout.toDouble)
  options.bidirectional().put(bidirectional)

  override private[torch] val nativeModule: LSTMImpl = LSTMImpl(options)
  nativeModule.to(paramType.toScalarType, false)

//T_TensorTensor_TOptional

  def apply(
             t: Tensor[ParamType],
             h0: Tensor[ParamType],
             c0: Tensor[ParamType]
           ): Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.native, c0.native)
    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx_opt)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), fromNative(fore2.get0()), fromNative(fore2.get1()))
  }  
  def apply(
      t: Tensor[ParamType],
      h0: Option[Tensor[ParamType]] = None,
      c0: Option[Tensor[ParamType]] = None
  ): Tuple3[Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx_opt)
    val fore2 = fore.get1()
    (fromNative(fore.get0()), fromNative(fore2.get0()), fromNative(fore2.get1()))
  }


  def forward_with_packed_input(packed_input: PackedSequence): PackedSequenceTensorTensor = {

    val output = nativeModule.forward_with_packed_input(packed_input)
    (output.get0(),fromNative(output.get1().get0()),fromNative(output.get1().get1()))

  }

  def forward_with_packed_input(packed_input: PackedSequence, hx: Tensor[ParamType], cx: Tensor[ParamType]): PackedSequenceTensorTensor = {
    val hxx = new T_TensorTensor_T(hx.native, cx.native)
    val hxx_opt = new T_TensorTensor_TOptional(hxx)
    val output = nativeModule.forward_with_packed_input(packed_input, hxx_opt)
    (output.get0(),fromNative(output.get1().get0()),fromNative(output.get1().get1()))
  }
  def weight: TensorVector = nativeModule.all_weights()

  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"${getClass.getSimpleName}(inputSize=$inputSize, hiddenSize=$hiddenSize,numLayers=${numLayers},batchFirst = ${batchFirst},dropout = ${dropout},bidirectional = ${bidirectional} bias=$bias)"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object LSTM:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      input_size: Int,
      hidden_size: Int,
      num_layers: Int,
      bias: Boolean = true,
      batch_first: Boolean = false,
      dropout: Float = 0.1f,
      bidirectional: Boolean = false
  ): LSTM[ParamType] =
    new LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)













//options.hidden_size().put(hiddenSize.toLong)
//options.num_layers().put(numLayers)

//  def apply(t: Tensor[ParamType],h0: Tensor[ParamType],c0: Tensor[ParamType]): (Tensor[ParamType],Tensor[ParamType],Tensor[ParamType]) = {
//    val fore = nativeModule.forward(t.native,h0.native,c0.native)
//    val fore2 = fore.get1()
//    (fromNative(fore.get0()),fromNative(fore2.get0()), fromNative(fore2.get1()))
//  }
