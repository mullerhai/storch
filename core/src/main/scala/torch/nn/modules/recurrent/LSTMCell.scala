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

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  LSTMCellImpl,
  LSTMCellOptions,
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
import torch.{Default, FloatNN, ComplexNN, Tensor}

/** Applies a 2D convolution over an input signal composed of several input planes. long input_size,
  * \@Cast("int64_t") long hidden_size
  * @group nn_conv
  */
final class LSTMCell[ParamType <: FloatNN | ComplexNN: Default](
    inputSize: Int,
    hiddenSize: Int,
    bias: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new LSTMCellOptions(inputSize.toLong, hiddenSize.toLong)
  options.hidden_size().put(hiddenSize.toLong)
  options.bias().put(bias)

  override private[torch] val nativeModule: LSTMCellImpl = LSTMCellImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  // T_TensorTensor_TOptional

  def apply(
             t: Tensor[ParamType],
             h0: Tensor[ParamType],
             c0: Tensor[ParamType]
           ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    val hx = new T_TensorTensor_T(h0.native, c0.native)
    val hx_opt = new T_TensorTensor_TOptional(hx)
    val fore = nativeModule.forward(t.native, hx_opt)
    (fromNative(fore.get0()), fromNative(fore.get1()))
  }
  def apply(
      t: Tensor[ParamType],
      h0: Option[Tensor[ParamType]] = None,
      c0: Option[Tensor[ParamType]] = None
  ): Tuple2[Tensor[ParamType], Tensor[ParamType]] = {
    if (h0.isDefined && c0.isDefined) {
      val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
      val hx_opt = new T_TensorTensor_TOptional(hx)
      val fore = nativeModule.forward(t.native, hx_opt)
      return  (fromNative(fore.get0()), fromNative(fore.get1()))
    }else {
//      val hx = new T_TensorTensor_T(h0.get.native, c0.get.native)
//      val hx_opt = new T_TensorTensor_TOptional(hx)
      val fore = nativeModule.forward(t.native)
      return  (fromNative(fore.get0()), fromNative(fore.get1()))
    }
 
  }

  def weight: TensorVector =
    TensorVector(nativeModule.weight_hh(), nativeModule.weight_ih()) /// all_weights()

  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"${getClass.getSimpleName}(inputSize=$inputSize, hiddenSize=$hiddenSize bias=$bias)"

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

object LSTMCell:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      input_size: Int,
      hidden_size: Int,
      bias: Boolean = true
  ): LSTMCell[ParamType] = new LSTMCell(input_size, hidden_size, bias)
 
 


 
 
 
 
 
 
 
 
 
 

//  def apply(t: Tensor[ParamType],h0: Tensor[ParamType],c0: Tensor[ParamType]): (Tensor[ParamType],Tensor[ParamType],Tensor[ParamType]) = {
//    val fore = nativeModule.forward(t.native,h0.native,c0.native)
//    val fore2 = fore.get1()
//    (fromNative(fore.get0()),fromNative(fore2.get0()), fromNative(fore2.get1()))
//  }
