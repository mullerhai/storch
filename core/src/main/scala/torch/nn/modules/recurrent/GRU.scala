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
  T_TensorT_TensorTensor_T_T,PackedSequence,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  GRUImpl,
  GRUOptions,
  TensorVector,
  kCircular,
  kReflect,
  kReplicate,
  kZeros
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.conv.Conv2d.PaddingMode

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class GRU[ParamType <: FloatNN | ComplexNN: Default](
    inputSize: Int,
    hiddenSize: Int,
    numLayers: Int,
    bias: Boolean = true,
    batchFirst: Boolean = false,
    dropout: Float = 0f,
    bidirectional: Boolean = false
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  type PackedSequenceTensor = (PackedSequence, Tensor[ParamType]) //T_PackedSequenceTensor_T
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new GRUOptions(inputSize.toLong, hiddenSize.toLong)
 
  options.input_size().put(LongPointer(1).put(inputSize.toLong))
  options.hidden_size().put(LongPointer(1).put(hiddenSize.toLong))
  options.num_layers().put(LongPointer(1).put(numLayers.toLong))
  options.bias().put(bias)
  options.batch_first().put(batchFirst)
  options.dropout().put(dropout.toDouble)
  options.bidirectional().put(bidirectional)

  override private[torch] val nativeModule: GRUImpl = GRUImpl(options)
  nativeModule.to(paramType.toScalarType, false)


  def apply(
             input: Tensor[ParamType],
             hx: Tensor[ParamType]
           ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore = nativeModule.forward(input.native, hx.native)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }
  
  def apply(
      input: Tensor[ParamType],
      hx: Option[Tensor[ParamType]] = None
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val fore = if hx.isDefined then nativeModule.forward(input.native, hx.get.native) else nativeModule.forward(input.native)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }

  
  
  def forward_with_packed_input(packed_input: PackedSequence): PackedSequenceTensor = {
    val output = nativeModule.forward_with_packed_input(packed_input)
    (output.get0(), fromNative(output.get1()))

  }

  def forward_with_packed_input(packed_input: PackedSequence, hx: Tensor[ParamType]): PackedSequenceTensor = {
    val output = nativeModule.forward_with_packed_input(packed_input, hx.native)
    (output.get0(), fromNative(output.get1()))

  }
  
  def weight: TensorVector = nativeModule.all_weights()

  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"${getClass.getSimpleName}(inputSize=$inputSize, hiddenSize=$hiddenSize,numLayers=${numLayers},batchFirst = ${batchFirst},dropout = ${dropout},bidirectional = ${bidirectional} bias=$bias)"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object GRU:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      input_size: Int,
      hidden_size: Int,
      num_layers: Int,
      bias: Boolean = true,
      batch_first: Boolean = false,
      dropout: Float = 0f,
      bidirectional: Boolean = false
  ): GRU[ParamType] =
    new GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)







//  def apply(
//             input: Tensor[ParamType]
//           ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
//    val fore = nativeModule.forward(input.native)
//    (fromNative(fore.get0()), fromNative(fore.get1()))
//  }  


//  def weight: TensorVector = fromNative(nativeModule.all_weights())
//  options.hidden_size().put(hiddenSize)