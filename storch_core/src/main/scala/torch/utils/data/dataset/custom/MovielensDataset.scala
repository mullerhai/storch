package torch.utils.data.dataset.custom

import org.bytedeco.pytorch.ExampleVector
import torch.utils.data.Dataset
import torch.{DType, Default, Int64, Tensor}

import java.nio.file.Paths

class MovielensDataset[Input <: DType, Target <: DType] extends Dataset[Input, Target] {

  val DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
  val DATA_DIR = "ml-1m"
  val DATA_PATH = Paths.get(DATA_DIR, "ratings.dat").toString
  val ZIP_PATH = "ml-1m.zip"

  override def length: Long = ???

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???

  override def features: Tensor[Input] = ???

  override def targets: Tensor[Target] = ???

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???

  override def get_batch(request: Long*): ExampleVector = ???

}

//  override def getItem(idx: Int): (Tensor[? <: DType], Tensor[Int64]) = ???
//
//  override def get_batch(request: Seq[Long]): ExampleVector =  super.get_batch(request*)
//
//  override def iterator: Iterator[(Tensor[? <: DType], Tensor[? <: DType])] = ???
