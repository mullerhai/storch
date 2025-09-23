package torch
package utils
package data
package dataloader
package distribute

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  Example,
  ExampleVector,
  ExampleIterator,
  ExampleVectorIterator,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  JavaDistributedSequentialDataLoader as DSDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.java.JavaDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.sampler.distribute.DistributedSequentialSampler
import torch.utils.data.dataset.java
import torch.utils.data.sampler
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import torch.utils.data.dataloader.TorchDataLoaderOptions

object DistributedSequentialDataLoader {

  def apply(
      dataset: JavaDataset,
      sampler: DistributedSequentialSampler,
      option: TorchDataLoaderOptions
  ) =
    new DistributedSequentialDataLoader(
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

class DistributedSequentialDataLoader(
    dataset: JavaDataset,
    sampler: DistributedSequentialSampler,
    batch_size: Int,
    shuffle: Boolean = false,
    num_workers: Int = 0,
    max_jobs: Long = 0L,
    drop_last: Boolean = false,
    in_order: Boolean = true,
    timeout: Float = 0
) extends DSDL(dataset, sampler, new DLOP())
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

  val nativeDataLoader = new DSDL(dataset, sampler, option.toNative)
  override def begin(): ExampleVectorIterator = nativeDataLoader.begin()

  override def end(): ExampleVectorIterator = nativeDataLoader.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = nativeDataLoader.options()

  override def iterator: Iterator[ExampleVector] = new Iterator[ExampleVector] {

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
