package torch
package utils
package data
package dataloader
package random

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ExampleIterator,
  ExampleVector,
  ExampleVectorIterator,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  ChunkBatchDataset as CBD,
  ChunkRandomDataLoader as CRDL,
  JavaRandomDataLoader as RDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataloader.random.RandomDataLoader
import torch.utils.data.dataloader.{TorchDataLoader, TorchDataLoaderOptions}
import torch.utils.data.dataset.normal.JavaDataset
import torch.utils.data.{NormalTensorDataset, sampler, Dataset as DatasetTrait}
import torch.utils.data.sampler.RandomSampler
import torch.{DType, Default}

object RandomDataLoader {

  def apply[ParamType <: DType: Default](
      dataset: JavaDataset | DatasetTrait[ParamType, ? <: DType],
      sampler: RandomSampler,
      option: TorchDataLoaderOptions
  ) =
    new RandomDataLoader(
      dataset,
      sampler,
      option.batch_size,
      option.shuffle,
      option.num_workers,
      option.max_jobs,
      option.drop_last,
      option.in_order,
      option.timeout
    )
}

class RandomDataLoader[ParamType <: DType: Default](
    dataset: JavaDataset | DatasetTrait[ParamType, ? <: DType],
    sampler: RandomSampler,
    batch_size: Int,
    shuffle: Boolean = false,
    num_workers: Int = 0,
    max_jobs: Long = 0L,
    drop_last: Boolean = false,
    in_order: Boolean = true,
    timeout: Float = 0
) extends RDL(dataset, sampler, new DLOP())
    with TorchDataLoader
    with Iterable[ExampleVector] {

  val option = TorchDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers,
    max_jobs = max_jobs,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout
  )

  private lazy val nativeDataLoader = new RDL(dataset, sampler, option.toNative)

  override def begin(): ExampleVectorIterator =
    nativeDataLoader.begin() // exampleVectorIterator //dataset.exampleVector.begin()

  override def end(): ExampleVectorIterator =
    nativeDataLoader.end() // exampleVectorIterator //dataset.exampleVector.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(
    option.toNative
  ) /// super.options()

  def getIteratorBuffer: mutable.Buffer[ExampleVector] = {
    val iteratorBuffer = new ListBuffer[ExampleVector]
    val nativeDataLoader = new RDL(dataset, sampler, option.toNative)
    var current: ExampleVectorIterator = nativeDataLoader.begin
    val endIterator: ExampleVectorIterator = nativeDataLoader.end
    while (!current.equals(endIterator)) {
      val example = current.access
      iteratorBuffer.append(example)
      current = current.increment()
    }
    iteratorBuffer
  }

  override def iterator: Iterator[ExampleVector] = getIteratorBuffer.iterator

  lazy val iteratorSeq: Seq[ExampleVector] = getIteratorBuffer.toSeq

  def iterator_raw: Iterator[ExampleVector] = new Iterator[ExampleVector] {

    private lazy val nativeDataLoader = new RDL(dataset, sampler, option.toNative)

    private var current: ExampleVectorIterator = nativeDataLoader.begin()

    private val endIterator: ExampleVectorIterator = nativeDataLoader.end()

    override def hasNext: Boolean = !current.equals(endIterator)

    override def next(): ExampleVector = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}

//  val exampleVectorIterator =ExampleVectorIterator(dataset.exampleVector) //.iterator()
//  val exampleVectorIterator =ExampleVectorIterator(dataset.get_batch()) //

//  val dataloader = new RDL(dataset,sampler,option)
