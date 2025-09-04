package torch
package optim

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

import torch.Tensor
import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{InputArchive, LossClosure, OptimizerOptions, OptimizerParamGroup, OptimizerParamGroupVector, OptimizerParamState, OutputArchive, TensorVector}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
//import org.bytedeco.pytorch.*

import scala.collection.immutable.{ArraySeq, SeqMap, TreeSeqMap}

case class TorchOptimizerParamGroup(paramGroup: OptimizerParamGroup) {
  val paramGroupDict: mutable.HashMap[String, Any] = new mutable.HashMap()
  _initDict
  private def _initDict = {
    if(paramGroup.has_options()){
      paramGroupDict.put("lr", paramGroup.options().get_lr())
    }
  }
}
/** Base class for all optimizers. */
abstract class Optimizer extends Pointer {
  private[torch] def native: pytorch.Optimizer

  val optimizerParamState: OptimizerParamState
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

  def parameters_raw(): TensorVector = native.parameters()

  def parameters(): Seq[Tensor[?]] = {
    val buffer = new ListBuffer[Tensor[?]]()
    val tensorVector = native.parameters()
    var it = tensorVector.begin()
    while(!it.equals(tensorVector.end())){
      buffer.append(from_native(it.get()))
      it = it.increment()
    }
    buffer.toSeq
  }
  def add_param_group(param_group: OptimizerParamGroup): Unit = native.add_param_group(param_group)

  def size(): Long = native.size()
  //   self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
  //        self.param_groups: List[Dict[str, Any]] = []
  //defaults: Dict[str, Any]
  def defaults(): OptimizerOptions = native.defaults()

  def get_optimizer_param_groups_raw(optimizer: Optimizer): Seq[OptimizerParamGroup] = {
    val groupBuffer = new ListBuffer[OptimizerParamGroup]()
    val optimizerParamGroupVector: OptimizerParamGroupVector = optimizer.native.param_groups()
    var element = optimizerParamGroupVector.begin()
    while (!element.equals(optimizerParamGroupVector.end())) {
      groupBuffer.append(element.get())
      element = element.increment()
    }
    groupBuffer.toSeq
  }

  def get_optimizer_param_groups(optimizer: Optimizer): Seq[TorchOptimizerParamGroup] = {
    val groupBuffer = new ListBuffer[TorchOptimizerParamGroup]()
    val optimizerParamGroupVector: OptimizerParamGroupVector = optimizer.native.param_groups()
    var element = optimizerParamGroupVector.begin()
    while (!element.equals(optimizerParamGroupVector.end())) {
      groupBuffer.append(TorchOptimizerParamGroup(element.get()))
      element = element.increment()
    }
    groupBuffer.toSeq
  }

  def param_groups: Seq[TorchOptimizerParamGroup] = {
    val groupBuffer = new ListBuffer[TorchOptimizerParamGroup]()
    val optimizerParamGroupVector: OptimizerParamGroupVector = native.param_groups()
    var element = optimizerParamGroupVector.begin()
    while (!element.equals(optimizerParamGroupVector.end())) {
      groupBuffer.append(TorchOptimizerParamGroup(element.get()))
      element = element.increment()
    }
    groupBuffer.toSeq
  }

  def paramGroups = this.param_groups

  def save(archive: OutputArchive): Unit = native.save(archive)

  def load(archive: InputArchive): Unit = native.load(archive)
}

//extension (group: OptimizerParamGroup)
//  def getDict: Map[String, Any] = {
//    // 实现逻辑
//    Map("param1" -> group, "param2" -> group)
//  }
//
//  def paramDict: Map[String, Any] = getDict // 计算属性
//  




///*
// * Copyright 2022 storch.dev
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package torch
//package optim
//
//import org.bytedeco.javacpp.Pointer
//import org.bytedeco.pytorch
//import org.bytedeco.pytorch.{
//  InputArchive,
//  LossClosure,
//  OptimizerOptions,
//  OptimizerParamGroup,
//  OptimizerParamGroupVector,
//  OutputArchive,
//  TensorVector
//}
//import scala.collection.immutable.{ArraySeq, SeqMap, TreeSeqMap}
//
///** Base class for all optimizers. */
//abstract class Optimizer extends Pointer {
//  private[torch] def native: pytorch.Optimizer
//
//  /** Performs a single optimization step (parameter update).
//    *
//    * @note
//    *   Unless otherwise specified, this function should not modify the ``.grad`` field of the
//    *   parameters.
//    */
//  def step(): Unit =
//    native.step()
//    // TODO check what tensor is returned by step
//    ()
//
//  def add_parameters(parameters: SeqMap[String, Tensor[?]]): Unit = {
//
//    val tensorVector = new TensorVector()
//    tensorVector.put(parameters.values.toArray.map(_.native)*)
//    native.add_parameters(tensorVector)
//
//  }
//
//  def add_parameters(parameters: Seq[Tensor[?]]): Unit = {
//
//    val tensorVector = new TensorVector()
//    tensorVector.put(parameters.map(_.native)*)
//    native.add_parameters(tensorVector)
//
//  }
//
//  def add_param_group[D <: DType](parameters: Seq[Tensor[D]], options: OptimizerOptions): Unit = {
//
//    val tensorVector = new TensorVector()
//    tensorVector.put(parameters.map(_.native)*)
//    val paramGroup = new OptimizerParamGroup(tensorVector, options)
//    native.add_param_group(paramGroup)
//
//  }
//  def add_param_group[D <: DType](
//      parameters: SeqMap[String, Tensor[D]],
//      options: OptimizerOptions
//  ): Unit = {
//    val tensorVector = new TensorVector()
//    tensorVector.put(parameters.values.toArray.map(_.native)*)
//    val paramGroup = new OptimizerParamGroup(tensorVector, options)
//    native.add_param_group(paramGroup)
//
//  }
//  def zeroGrad(setToNone: Boolean = true): Unit = native.zero_grad(setToNone)
//
//  /** Sets the gradients of all optimized `Tensor`s to zero. */
//  def zeroGrad(): Unit = native.zero_grad()
//
//  def step(closure: LossClosure): pytorch.Tensor = native.step(closure)
//
//  def zero_grad(set_to_none: Boolean): Unit = native.zero_grad(set_to_none)
//
//  def zero_grad(): Unit = native.zero_grad()
//
//  def add_parameters(parameters: TensorVector): Unit = native.add_parameters(parameters)
//
//  def parameters(): TensorVector = native.parameters()
//  def add_param_group(param_group: OptimizerParamGroup): Unit = native.add_param_group(param_group)
//
//  def size(): Long = native.size()
//
//  def defaults(): OptimizerOptions = native.defaults()
//
//  def param_groups(): OptimizerParamGroupVector = native.param_groups()
//
//  def save(archive: OutputArchive): Unit = native.save(archive)
//
//  def load(archive: InputArchive): Unit = native.load(archive)
//}
