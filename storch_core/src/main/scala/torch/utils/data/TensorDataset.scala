package torch.utils.data

import torch.{DType, Default, Tensor}

trait TensorDataset[ParamType <: DType: Default] {
//  def init(data: AnyRef*): Unit
  def length: Long
  def getItem(idx: Int): Tensor[ParamType]

//  def apply(data: AnyRef*): Unit =
//    init(data)
}
