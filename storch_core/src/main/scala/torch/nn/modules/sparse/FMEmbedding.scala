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
package sparse

import org.bytedeco.pytorch
import org.bytedeco.pytorch.EmbeddingImpl
import org.bytedeco.pytorch.TransformerImpl
import org.bytedeco.pytorch.MultiheadAttentionImpl
import org.bytedeco.pytorch.ModuleListImpl
import org.bytedeco.pytorch.SequentialImpl
import org.bytedeco.pytorch.EmbeddingOptions
import torch.internal.NativeConverters.{fromNative, toNative}

// format: off
/** A simple lookup table that stores embeddings of a fixed dictionary and size.
 *
 * This module is often used to store word embeddings and retrieve them using indices. The input to
 * the module is a list of indices, and the output is the corresponding word embeddings.
 *
 * @group nn_sparse
 *
 * @param numEmbeddings
 *   Size of the dictionary of embeddings
 * @param embeddingDim
 *   The size of each embedding vector
 * @param paddingIdx
 *   If specified, the entries at `paddingIdx` do not contribute to the gradient; therefore, the
 *   embedding vector at `paddingIdx` is not updated during training, i.e. it remains as a fixed
 *   "pad". For a newly constructed Embedding, the embedding vector at `paddingIdx` will default to
 *   all zeros, but can be updated to another value to be used as the padding vector.
 * @param maxNorm
 *   If given, each embedding vector with norm larger than `maxNorm` is renormalized to have norm
 *   `maxNorm`.
 * @param normType
 *   The p of the p-norm to compute for the `maxNorm` option. Default `2`.
 * @param scaleGradByFreq
 *   If given, this will scale gradients by the inverse of frequency of the words in the
 *   mini-batch. Default `false`.
 * @param sparse
 *   If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor. See Notes for more
 *   details regarding sparse gradients.
 *
 * @see See [[https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_embedding.html#class-embedding Pytorch C++ Embedding]]
 * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html Pytorch Python Embedding]]
 */
// format: on
final class FMEmbedding[ParamType <: FloatNN | ComplexNN: Default](
    numEmbeddings: Int,
    embeddingDim: Int,
    paddingIdx: Option[Int] = None,
    maxNorm: Option[Float] = None,
    normType: Option[Float] = Some(2.0f),
    scaleGradByFreq: Boolean = false,
    sparse: Boolean = false
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new EmbeddingOptions(numEmbeddings.toLong, embeddingDim.toLong)
  paddingIdx.foreach(p => options.padding_idx().put(toNative(p)))
  maxNorm.foreach(m => options.max_norm().put(m))
  normType.foreach(n => options.norm_type().put(n))
  options.scale_grad_by_freq().put(scaleGradByFreq)
  options.sparse().put(sparse)

  override val nativeModule: EmbeddingImpl = EmbeddingImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  override def hasBias(): Boolean = false

  def weight: Tensor[ParamType] = fromNative(nativeModule.weight)
  def weight_=(w: Tensor[ParamType]): Tensor[ParamType] =
    nativeModule.weight(w.native)
    w

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  def apply(indices: Tensor[Int64 | Int32], weight: Option[Tensor[ParamType]] = None): Tensor[ParamType] = indices match{
    case input : Tensor[Int64] =>  fromNative(nativeModule.forward(indices.native))
    case input : Tensor[Int32] => fromNative(nativeModule.forward(indices.to(torch.int64).native))
  }

//  def apply(indices: Tensor[Int64]|Tensor[Int32], weight: Option[Tensor[ParamType]] = None): Tensor[ParamType] =
//    indices match
//      case torch.int64 => fromNative(nativeModule.forward(indices.native))
//      case torch.int32 => fromNative(nativeModule.forward(indices.to(torch.int64).native))

  
  
  override def toString(): String =
    val numEmbed = s"numEmbeddings=$numEmbeddings"
    val dim = s"embeddingDim=$embeddingDim"
    val padding = s"paddingIdx=$paddingIdx"
    val maxN = s"maxNorm=$maxNorm"
    val normT = s"normType=$normType"
    val scale = s"scaleGradByFreq=$scaleGradByFreq"
    val s = s"sparse=$sparse"
    s"${getClass().getSimpleName()}($numEmbed, $dim, $padding, $maxN, $normT, $scale, $s )"

object FMEmbedding:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      num_embeddings: Int,
      embedding_dim: Int,
      padding_idx: Option[Int] = None,
      max_norm: Option[Float] = None,
      norm_type: Option[Float] = Some(2.0f),
      scale_grad_by_freq: Boolean = false,
      sparse: Boolean = false
  ): FMEmbedding[ParamType] = new FMEmbedding(
    num_embeddings,
    embedding_dim,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse
  )

//  def apply(t: Tensor[Int64]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))
