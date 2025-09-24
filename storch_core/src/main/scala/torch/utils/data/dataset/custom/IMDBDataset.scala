package torch
package utils
package data.dataset.custom

import org.bytedeco.pytorch.ExampleVector
import torch.utils.data.Dataset
import torch.{DType, Default, Int64, Tensor}

import java.nio.file.Paths

class IMDBDataset[Input <: DType, Target <: DType] extends Dataset[Input, Target] {

  val DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  val DATA_ROOT = "./data"
  val tarGzPath = Paths.get(DATA_ROOT, "aclImdb_v1.tar.gz")
  val dataDir = Paths.get(DATA_ROOT, "aclImdb")

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
