package torch.utils.data.dataset.custom

import org.bytedeco.pytorch.ExampleVector
import torch.utils.data.Dataset
import torch.{DType, Default, Int64, Tensor}

class NetflixPrizeDataset[Input <: DType, Target <: DType] extends Dataset[Input, Target] {

  override def length: Long = ???

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???

  override def features: Tensor[Input] = ???

  override def targets: Tensor[Target] = ???

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???

  override def get_batch(request: Long*): ExampleVector = ???

}

