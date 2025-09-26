package torch.utils.data.dataset.custom

import org.bytedeco.pytorch.ExampleVector
import torch.utils.data.Dataset
import torch.{DType, Default, Int64, Tensor}

class AmazonDataset[Input <: DType, Target <: DType] extends Dataset[Input, Target] {

  val DATA_URL =
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
  val DATA_PATH = "reviews_Electronics_5.json.gz"

  override def length: Long = ???

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???

  override def features: Tensor[Input] = ???

  override def targets: Tensor[Target] = ???

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???

  override def get_batch(request: Long*): ExampleVector = ??? // super.get_batch(request)
}

