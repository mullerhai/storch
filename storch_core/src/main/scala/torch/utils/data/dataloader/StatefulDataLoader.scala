package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaStatefulDataset,
  JavaStatefulDataLoader as SDL,
  DataLoaderOptions,
  ExampleVectorIterator,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  TensorVector,
  Example,
  ExampleVector
}
import torch.utils.data.dataset.java.StatefulDataset
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import torch.utils.data.dataloader.TorchDataLoaderOptions
import torch.utils.data.Dataset as DatasetTrait

object StatefulDataLoader {
  
  def apply(dataset: StatefulDataset, option: TorchDataLoaderOptions) =
    new StatefulDataLoader(dataset, option.batch_size, option.shuffle, option.num_workers, option.max_jobs, option.drop_last, option.in_order, option.timeout)
}


class StatefulDataLoader(dataset: StatefulDataset,
                         batch_size: Int,
                         shuffle: Boolean = false,
                         num_workers: Int = 0,
                         max_jobs: Long = 0l,
                         drop_last: Boolean = false,
                         in_order: Boolean = true,
                         timeout: Float = 0
                        )
    extends SDL(dataset, new DLOP())
    with TorchDataLoader with Iterable[ExampleVector] {

  val option = TorchDataLoaderOptions(batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, max_jobs = max_jobs, drop_last = drop_last, in_order = in_order, timeout = timeout)

  val nativeDataLoader = new SDL(dataset, option.toNative)

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
