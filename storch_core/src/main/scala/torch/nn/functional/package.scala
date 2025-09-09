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

import org.bytedeco.javacpp.annotation.{Const, ByRef, ByVal, Namespace}

/** @groupname nn_conv Convolution functions
  * @groupname nn_pooling Pooling functions
  * @groupname nn_attention Attention mechanisms
  * @groupname nn_activation Non-linear activation functions
  * @groupname nn_linear Linear functions
  * @groupname nn_dropout Dropout functions
  * @groupname nn_sparse Sparse functions
  * @groupname nn_distance Distance functions
  * @groupname nn_loss Loss functions
  * @groupname nn_vision Vision functions
  */
package object functional
    extends Activations
    with BatchNorm
    with Convolution
    with Distance
    with Dropout
    with Linear
    with Loss
    with Normalization
    with Linalg
    with Pooling
    with Sparse
    with UnSampling
    with Vision
    with FFT
    with Fold
    with Padding
    with Recurrent

//@native
//@Namespace("at::native")
//@ByVal def log[TT <: DType](@Const @ByRef self: Tensor[TT]): Tensor[TT]

//@native
//@Namespace("at::native")
//@ByVal def view[TT <: DType](@Const @ByRef self: Tensor[ TT], @ByVal size: IntArrayRef): Tensor[ TT]

//@native
//@Namespace("at::native") @ByVal def relu[TT <: DType](@Const @ByRef input: Tensor[TT]): Tensor[ TT]

//@native
//@Namespace("at::native")
//@ByVal def log_softmax[TT <: DType](@Const @ByRef self: Tensor[TT], @Cast(Array("int64_t")) dim: CLongPointer): Tensor[TT]
//
//@native
//@Namespace("at::native")
//@ByVal def linear[TT <: DType](@Const @ByRef input: Tensor[TT], @Const @ByRef weight: Tensor[TT], @Const @ByRef bias: Tensor[TT] | Option[Tensor[TT]]): Tensor[TT]
//
//
//@native
//@Namespace("at::native")
//@ByVal def view[T, TT <: DType](@Const @ByRef self: Tensor[ TT], @ByVal size: IntArrayRef): Tensor[ TT]

//@native
//@Namespace("at::native")
//@ByVal def mkldnn_view[T, TT <: DType](@Const @ByRef self: Tensor[TT], @ByVal size: IntArrayRef): Tensor[ TT]
////@native
//@Namespace("at::native")
//@ByVal def lstm[TT <: DType](@Const @ByRef input: Tensor[TT], @ByVal hx: TensorList[TT], @ByVal params: TensorList[TT], @Cast(Array("bool")) has_biases: Boolean, @Cast(Array("int64_t")) num_layers: CLongPointer, dropout: Double, @Cast(Array("bool")) train: Boolean, @Cast(Array("bool")) bidirectional: Boolean, @Cast(Array("bool")) batch_first: Boolean): TensorTriple[T, T, T, TT]
//
//@native
//@Namespace("at::native")
//@ByVal def lstm[T, TT <: DType](@Const @ByRef input: Tensor[T, TT], @ByVal hx: TensorList[T, TT], @ByVal params: TensorList[T, TT], @Cast(Array("bool")) has_biases: Boolean, @Cast(Array("int64_t")) num_layers: CLongPointer, dropout: Double, @Cast(Array("bool")) train: Boolean, @Cast(Array("bool")) bidirectional: Boolean, @Cast(Array("bool")) batch_first: Boolean): TensorTriple[T, T, T, TT]
//
//@native
//@Namespace("at::native")
//@ByVal def lstm[T, TT <: DType](@Const @ByRef data: Tensor[T, TT], @Const @ByRef batch_sizes: Tensor[T, TT], @ByVal hx: TensorList[T, TT], @ByVal params: TensorList[T, TT], @Cast(Array("bool")) has_biases: Boolean, @Cast(Array("int64_t")) num_layers: CLongPointer, dropout: Double, @Cast(Array("bool")) train: Boolean, @Cast(Array("bool")) bidirectional: Boolean): TensorTriple[T, T, T, TT]
//}
