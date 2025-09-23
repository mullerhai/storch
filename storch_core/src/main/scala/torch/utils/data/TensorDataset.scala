package torch.utils.data

import torch.{DType, Default, Tensor}

trait TensorDataset {

  def length: Long
  
  def getItem(idx: Int): Tensor[? <: DType]
  
}

















//  def init(data: AnyRef*): Unit
//  def apply(data: AnyRef*): Unit =
//    init(data)