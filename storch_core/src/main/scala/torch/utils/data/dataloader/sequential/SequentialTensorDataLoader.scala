package torch
package utils
package data
package dataloader
package sequential

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  FullDataLoaderOptions,
  TensorExample,
  TensorExampleVector,
  TensorExampleIterator,
  TensorExampleVectorIterator,
  DataLoaderOptions as DLOP,
  JavaSequentialTensorDataLoader as STDL
}
import torch.utils.data.dataset.normal
import torch.utils.data.dataset.normal.NormalTensorDataset
import torch.utils.data.sampler
import torch.utils.data.sampler.SequentialSampler
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object SequentialTensorDataLoader {

  def apply(
      dataset: normal.NormalTensorDataset,
      sampler: SequentialSampler,
      option: TorchTensorDataLoaderOptions
  ) =
    new SequentialTensorDataLoader(
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

class SequentialTensorDataLoader(
    dataset: normal.NormalTensorDataset,
    sampler: SequentialSampler,
    batch_size: Int,
    shuffle: Boolean = false,
    num_workers: Int = 0,
    max_jobs: Long = 0L,
    drop_last: Boolean = false,
    in_order: Boolean = true,
    timeout: Float = 0
) extends STDL(dataset, sampler, new DLOP())
    with TorchDataLoader
    with Iterable[TensorExampleVector] {

  val option = TorchTensorDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers,
    max_jobs = max_jobs,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout
  )

  private lazy val nativeDataLoader = new STDL(dataset, sampler, option.toNative)

  override def begin(): TensorExampleVectorIterator = nativeDataLoader.begin()

  override def end(): TensorExampleVectorIterator = nativeDataLoader.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = nativeDataLoader.options()

  def getIteratorBuffer: mutable.Buffer[TensorExampleVector] = {
    val iteratorBuffer = new ListBuffer[TensorExampleVector]
    val nativeDataLoader = new STDL(dataset, sampler, option.toNative)
    var current: TensorExampleVectorIterator = nativeDataLoader.begin
    val endIterator: TensorExampleVectorIterator = nativeDataLoader.end
    while (!current.equals(endIterator)) {
      val example = current.access
      iteratorBuffer.append(example)
      current = current.increment()
    }
    iteratorBuffer
  }

  override def iterator: Iterator[TensorExampleVector] = getIteratorBuffer.iterator

  lazy val iteratorSeq: Seq[TensorExampleVector] = getIteratorBuffer.toSeq

  def iterator_raw: Iterator[TensorExampleVector] = new Iterator[TensorExampleVector] {

    private lazy val nativeDataLoader = new STDL(dataset, sampler, option.toNative)

    private var current: TensorExampleVectorIterator = nativeDataLoader.begin

    private val endIterator: TensorExampleVectorIterator = nativeDataLoader.end

    override def hasNext: Boolean = !current.equals(endIterator)

    override def next(): TensorExampleVector = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}
