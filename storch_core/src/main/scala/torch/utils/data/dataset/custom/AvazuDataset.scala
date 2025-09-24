package torch.utils.data.dataset.custom

import org.bytedeco.pytorch.ExampleVector
import torch.utils.data.Dataset
import torch.{DType, Default, Int64, Tensor}

class AvazuDataset[Input <: DType, Target <: DType] extends Dataset[Input, Target] {

  override def length: Long = ???

  override def features: Tensor[Input] = ???

  override def targets: Tensor[Target] = ???

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???

  override def get_batch(request: Long*): ExampleVector = ???

}

//  override def getItem(idx: Int): (Tensor[? <: DType], Tensor[Int64]) = ???
//
//  override def get_batch(request: Seq[Long]): ExampleVector =  super.get_batch(request*)
//
//  override def iterator: Iterator[(Tensor[? <: DType], Tensor[? <: DType])] = ???

//  override def get(index: Long): Example = super.get(index)

//  override def get_batch(request: Seq[Long]): ExampleVector = super.get_batch(request)

//  override def get_batch(indices: SizeTArrayRef): ExampleVector = super.get_batch(indices)
//  override def get_batch(indices: Seq[Int]): (Tensor[ParamType], Tensor[Int64]) = ???

//  override def get_batch(indices: SizeTArrayRef): ExampleVector = super.get_batch(indices)

//= super.iterator
