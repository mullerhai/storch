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
package optim

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  InputArchive,
  LossClosure,
  OptimizerOptions,
  OptimizerParamGroup,
  OptimizerParamGroupVector,
  OutputArchive,
  TensorVector
}
import scala.collection.immutable.{ArraySeq, SeqMap, TreeSeqMap}

/** Base class for all optimizers. */
abstract class Optimizer extends Pointer {
  private[torch] def native: pytorch.Optimizer

  /** Performs a single optimization step (parameter update).
    *
    * @note
    *   Unless otherwise specified, this function should not modify the ``.grad`` field of the
    *   parameters.
    */
  def step(): Unit =
    native.step()
    // TODO check what tensor is returned by step
    ()

  def add_parameters(parameters: SeqMap[String, Tensor[?]]): Unit = {

    val tensorVector = new TensorVector()
    tensorVector.put(parameters.values.toArray.map(_.native)*)
    native.add_parameters(tensorVector)

  }

  def add_parameters(parameters: Seq[Tensor[?]]): Unit = {

    val tensorVector = new TensorVector()
    tensorVector.put(parameters.map(_.native)*)
    native.add_parameters(tensorVector)

  }

  def add_param_group[D <: DType](parameters: Seq[Tensor[D]], options: OptimizerOptions): Unit = {

    val tensorVector = new TensorVector()
    tensorVector.put(parameters.map(_.native)*)
    val paramGroup = new OptimizerParamGroup(tensorVector, options)
    native.add_param_group(paramGroup)

  }
  def add_param_group[D <: DType](
      parameters: SeqMap[String, Tensor[D]],
      options: OptimizerOptions
  ): Unit = {
    val tensorVector = new TensorVector()
    tensorVector.put(parameters.values.toArray.map(_.native)*)
    val paramGroup = new OptimizerParamGroup(tensorVector, options)
    native.add_param_group(paramGroup)

  }
  def zeroGrad(setToNone: Boolean = true): Unit = native.zero_grad(setToNone)

  /** Sets the gradients of all optimized `Tensor`s to zero. */
  def zeroGrad(): Unit = native.zero_grad()

  def step(closure: LossClosure): pytorch.Tensor = native.step(closure)

  def zero_grad(set_to_none: Boolean): Unit = native.zero_grad(set_to_none)

  def zero_grad(): Unit = native.zero_grad()

  def add_parameters(parameters: TensorVector): Unit = native.add_parameters(parameters)

  def parameters(): TensorVector = native.parameters()
  def add_param_group(param_group: OptimizerParamGroup): Unit = native.add_param_group(param_group)

  def size(): Long = native.size()

  def defaults(): OptimizerOptions = native.defaults()

  def param_groups(): OptimizerParamGroupVector = native.param_groups()

  def save(archive: OutputArchive): Unit = native.save(archive)

  def load(archive: InputArchive): Unit = native.load(archive)
}
