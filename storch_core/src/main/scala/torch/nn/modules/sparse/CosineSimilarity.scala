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
import org.bytedeco.pytorch.{
  CosineSimilarityImpl,
  CosineSimilarityOptions,
  ModuleListImpl,
  PairwiseDistanceImpl,
  PairwiseDistanceOptions,
  SequentialImpl,
  TransformerImpl
}
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
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
 *   
 *   val m12 = nn.CosineSimilarity(dim = 1, eps = 1e-6)
 *   val input1 = torch.randn(Seq(100, 128))
 *   val input2 = torch.randn(Seq(100, 128))
 *   assertEquals(m12(input1, input2).shape, Seq(100)) // torch.Size([100]) //错误得到128
 *   
 * @see See [[https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_embedding.html#class-embedding Pytorch C++ Embedding]]
 * @see See [[https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html Pytorch Python Embedding]]
 */
// format: on
final class CosineSimilarity[ParamType <: FloatNN | ComplexNN: Default](
    dim: Int | Option[Int] = 1,
    eps: Double | Option[Double] = 1e-8
) extends HasParams[ParamType]
    with HasWeight[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options = new CosineSimilarityOptions()
  dim match {
    case px: Int => options.dim().put(LongPointer(1).put(px.toLong))
    case px: Option[Int] =>
      if px.isDefined then options.dim().put(LongPointer(1).put(px.get.toLong))
  }
  eps match {
    case ep: Double => options.eps().put(DoublePointer(1).put(ep.toDouble))
    case ep: Option[Double] =>
      if ep.isDefined then options.eps().put(DoublePointer(1).put(ep.get.toDouble))
  }

  override val nativeModule: CosineSimilarityImpl = CosineSimilarityImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()
  def apply(input1: Tensor[ParamType], input2: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input1.native, input2.native)
  )
  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
  override def toString(): String =
    s"${getClass().getSimpleName()}  dim = ${dim} eps = ${eps} )"

  override def weight: Tensor[ParamType] = ???

object CosineSimilarity:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      dim: Int | Option[Int] = 1,
      eps: Double | Option[Double] = 1e-8
  ): CosineSimilarity[ParamType] = new CosineSimilarity(dim, eps)
