package torch
package utils
package trainer

import torch.nn
import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.loss.LossFunc
import torch.nn.{Dropout, Embedding, Linear, Transformer, functional as F}
import torch.::
import torch.optim.Optimizer
import torch.utils.data.{Dataset, DataLoader}
import java.nio.file.Paths
import scala.math.*

trait Trainable[ParamType <: FloatNN: Default,  M <: TensorModule[ParamType] with HasParams[ParamType]](
                                                                             model: M,
                                                                             train_loader: DataLoader[ParamType],
                                                                             eval_loader: DataLoader[ParamType],
                                                                             optimizer_type: String = "adamW",
                                                                             criterion: LossFunc,
                                                                             device: Device = Device.CPU
                                                                           ) :

  def train(num_epochs: Int, learning_rate: Float = 0.001f, batch_size: Int, eval_interval: Int = 200): M


