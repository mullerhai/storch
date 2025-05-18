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
import org.bytedeco.pytorch.EmbeddingBagImpl
import org.bytedeco.pytorch.TransformerImpl
import org.bytedeco.pytorch.MultiheadAttentionImpl
import org.bytedeco.pytorch.ModuleListImpl
import org.bytedeco.pytorch.SequentialImpl
import org.bytedeco.pytorch.{kSum, kMean, kMax}
import org.bytedeco.pytorch.EmbeddingBagOptions
import torch.nn.modules.sparse.EmbeddingBag.EmbeddingBagMode
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.pytorch.global.torch.ScalarType
// format: off
/** A simple lookup table that stores embeddings of a fixed dictionary and size.
 *
 * This module is often used to store word embeddings and retrieve them using indices. The input to
 * the module is a list of indices, and the output is the corresponding word embeddings.
 *
 * @group nn_sparse
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
 * @see See [[https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_embedding.html#class-embedding Pytorch C++ Embedding]]
 * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html Pytorch Python Embedding]]
 * @native private def allocate(@Cast(Array("int64_t"))  num_embeddings: Long, @Cast(Array("int64_t"))  embedding_dim: Long): Unit
 * @Cast(Array("int64_t*")) @ByRef  @NoException(true)  @native  def num_embeddings: LongPointer
 * @Cast(Array("int64_t*")) @ByRef  @NoException(true)  @native  def embedding_dim: LongPointer
 * @ByRef @NoException(true)  @native  def max_norm: DoubleOptional
 * @ByRef @NoException(true)  @native  def norm_type: DoublePointer
 * @Cast(Array("bool*")) @ByRef  @NoException(true)  @native  def scale_grad_by_freq: BoolPointer
 * @ByRef @NoException(true)  @native  def mode: EmbeddingBagMode
 * @Cast(Array("bool*")) @ByRef  @NoException(true)  @native  def sparse: BoolPointer
 * @ByRef @NoException(true)  @native  def _weight: Tensor
 * @Cast(Array("bool*")) @ByRef  @NoException(true)  @native  def include_last_offset: BoolPointer
 * @ByRef @NoException(true)  @native  def padding_idx: LongOptional
 *        }
 *        torch.nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=None,
 *        norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False,
 *        _weight=None,
 *        include_last_offset=False,
 *        padding_idx=None, device=None, dtype=None)
 */
// format: on
final class EmbeddingBag[ParamType <: FloatNN | ComplexNN: Default](
    numEmbeddings: Int,
    embeddingDim: Int,
    maxNorm: Option[Float] | Float = None,
    normType: Option[Float] | Float = Some(2.0f),
    scaleGradByFreq: Boolean | Option[Boolean] = false,
    mode: EmbeddingBagMode | String = EmbeddingBagMode.kMean,
    sparse: Boolean | Option[Boolean] = false,
    includeLastOffset: Boolean | Option[Boolean] = false,
    paddingIdx: Option[Int] | Int = None,
    needWeight: Option[Tensor[ParamType]] = None
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new EmbeddingBagOptions(numEmbeddings.toLong, embeddingDim.toLong)

  maxNorm match {
    case m: Float => options.max_norm().put(m.toDouble)
    case m: Option[Float] => if m.isDefined then options.max_norm().put(m.get.toDouble)
  }

  normType match {
    case n: Float => options.norm_type().put(n.toDouble)
    case n: Option[Float] => if n.isDefined then options.norm_type().put(n.get.toDouble)
  }
  scaleGradByFreq match {
    case s : Boolean => options.scale_grad_by_freq().put(s)
    case s : Option[Boolean] => if s.isDefined then options.scale_grad_by_freq().put(s.get)
  }
  sparse match {
    case s : Boolean => options.sparse().put(s)
    case s : Option[Boolean] => if s.isDefined then options.sparse().put(s.get)
  }
  includeLastOffset match {
    case i : Boolean => options.include_last_offset().put(i)
    case i : Option[Boolean] => if i.isDefined then options.include_last_offset().put(i.get)
  }

  paddingIdx match {
    case p : Int => options.padding_idx().put(p)
    case p : Option[Int] =>if p.isDefined then options.padding_idx().put(p.get)
  }


  if needWeight.isDefined then options._weight().put(needWeight.get.native)


  mode match
    case EmbeddingBagMode.kMean | "mean" | "Mean" => options.mode().put(new kMean)
    case EmbeddingBagMode.kMax | "max" | "Max"    => options.mode().put(new kMax)
    case EmbeddingBagMode.kSum | "sum" | "Sum"    => options.mode().put(new kSum)

  override val nativeModule: EmbeddingBagImpl = EmbeddingBagImpl(options)
  nativeModule.to(paramType.toScalarType, false)
  override def hasBias(): Boolean = false
  def weight: Tensor[ParamType] = fromNative(nativeModule.weight)
  def weight_=(w: Tensor[ParamType]): Tensor[ParamType] =
    nativeModule.weight(w.native)
    w

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native.to(ScalarType.Long))
  )

  def apply(indices: Tensor[Int64]|Tensor[Int32], weight: Option[Tensor[ParamType]] = None): Tensor[ParamType] = {
    indices.dtype match
      case torch.int64 => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))
      case torch.int32 => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))
  }

  //    case input : Tensor[Int64] => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))
  //    case input : Tensor[Int32] => fromNative(nativeModule.forward(indices.native.to(ScalarType.Long)))

//  def apply(indices: Tensor[Int32 |Int64], weight: Option[String] = None,word:Option[String] = None,seq:Int =0): Tensor[ParamType] = fromNative(
//    nativeModule.forward(indices.native.to(ScalarType.Long))
//  )

  def apply(input: Tensor[ParamType], offsets: Tensor[Int64]|Tensor[Int32], size: Seq[Int]): Tensor[ParamType] = {
    fromNative(
      nativeModule.forward(
        input.native.to(ScalarType.Long),
        offsets.native.to(ScalarType.Long),
        torch.empty(size).native.to(ScalarType.Float)
      )
    )
  }

  def apply32(
      input: Tensor[ParamType],
      offsets: Tensor[Int32],
      per_sample_weights: Tensor[ParamType]
  ): Tensor[ParamType] = {
    fromNative(
      nativeModule.forward(
        input.native.to(ScalarType.Long),
        offsets.native.to(ScalarType.Long),
        per_sample_weights.native.to(ScalarType.Float)
      )
    )
  }

  def apply(
      input: Tensor[ParamType],
      offsets: Tensor[Int64]|Tensor[Int32],
      per_sample_weights: Tensor[ParamType]
  ): Tensor[ParamType] = {
    fromNative(
      nativeModule.forward(
        input.native.to(ScalarType.Long),
        offsets.native.to(ScalarType.Long),
        per_sample_weights.native.to(ScalarType.Float)
      )
    )
  }

  override def toString(): String =
    val numEmbed = s"numEmbeddings=$numEmbeddings"
    val dim = s"embeddingDim=$embeddingDim"
    val padding = s"paddingIdx=$paddingIdx"
    val maxN = s"maxNorm=$maxNorm"
    val normT = s"normType=$normType"
    val scale = s"scaleGradByFreq=$scaleGradByFreq"
    val s = s"sparse=$sparse"
    s"${getClass().getSimpleName()}($numEmbed, $dim, $padding, $maxN, $normT, $scale, $s )"

object EmbeddingBag:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      num_embeddings: Int,
      embedding_dim: Int,
      max_norm: Option[Float] | Float = None,
      norm_type: Option[Float] | Float = Some(2.0f),
      scale_grad_by_freq: Boolean | Option[Boolean] = false,
      mode: EmbeddingBagMode | String = EmbeddingBagMode.kMean,
      sparse: Boolean | Option[Boolean] = false,
      include_last_offset: Boolean | Option[Boolean] = false,
      padding_idx: Option[Int] | Int = None,
      need_weight: Option[Tensor[ParamType]] = None
  ): EmbeddingBag[ParamType] = new EmbeddingBag(
    num_embeddings,
    embedding_dim,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    mode,
    sparse,
    include_last_offset,
    padding_idx,
    need_weight
  )
  enum EmbeddingBagMode:
    case kSum, kMean, kMax























//  paddingIdx.foreach(p => options.padding_idx().put(toNative(p)))
//  maxNorm.foreach(m => options.max_norm().put(m))
//  normType.foreach(n => options.norm_type().put(n))
//  options.mode().put(modeNative)







//if maxNorm.isDefined then options.max_norm().put(maxNorm.get.toDouble)
//if normType.isDefined then options.norm_type().put(normType.get.toDouble)
//options.scale_grad_by_freq().put(scaleGradByFreq)
//options.sparse().put(sparse)
//options.include_last_offset().put(includeLastOffset)
//if paddingIdx.isDefined then options.padding_idx().put(paddingIdx.get)