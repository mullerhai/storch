package torch
package utils
package data
package dataloader
package distribute

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  TensorExample,
  TensorExampleIterator,
  TensorExampleVectorIterator,
  JavaDistributedSequentialTensorDataLoader as DSTDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.java.NormalTensorDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.sampler.distribute.DistributedSequentialSampler
import torch.utils.data.dataset.java
import torch.utils.data.sampler
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import torch.utils.data.dataloader.TorchTensorDataLoaderOptions

object DistributedSequentialTensorDataLoader {
  def apply(
      dataset: NormalTensorDataset,
      sampler: DistributedSequentialSampler,
      option: TorchTensorDataLoaderOptions
  ) =
    new DistributedSequentialTensorDataLoader(
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

class DistributedSequentialTensorDataLoader(
    dataset: NormalTensorDataset,
    sampler: DistributedSequentialSampler,
    batch_size: Int,
    shuffle: Boolean = false,
    num_workers: Int = 0,
    max_jobs: Long = 0L,
    drop_last: Boolean = false,
    in_order: Boolean = true,
    timeout: Float = 0
) extends DSTDL(dataset, sampler, new DLOP())
    with TorchDataLoader
    with Iterable[TensorExample] {

  val option = TorchTensorDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_workers,
    max_jobs = max_jobs,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout
  )

  val nativeDataLoader = new DSTDL(dataset, sampler, option.toNative)

  override def begin(): TensorExampleVectorIterator = nativeDataLoader.begin()

  override def end(): TensorExampleVectorIterator = nativeDataLoader.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(option.toNative)

  override def iterator: Iterator[TensorExample] = new Iterator[TensorExample] {

    private var current: TensorExampleIterator =
      nativeDataLoader.begin.asInstanceOf[TensorExampleIterator]

    private val endIterator: TensorExampleIterator =
      nativeDataLoader.end.asInstanceOf[TensorExampleIterator]

    override def hasNext: Boolean = !current.equals(endIterator)

    override def next(): TensorExample = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}
