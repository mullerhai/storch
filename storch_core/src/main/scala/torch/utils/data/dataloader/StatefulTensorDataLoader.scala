package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import org.bytedeco.pytorch.{FullDataLoaderOptions, SizeTOptional, TensorExample, TensorExampleIterator, TensorExampleVectorIterator, JavaStatefulTensorDataLoader as STDL, RandomSampler as RS, SequentialSampler as SS}
import torch.utils.data.dataset.java.StatefulTensorDataset
import torch.utils.data.dataset.java

object StatefulTensorDataLoader {
  def apply(dataset: java.StatefulTensorDataset, option: TorchTensorDataLoaderOptions) =
    new StatefulTensorDataLoader(dataset, option.batch_size, option.shuffle, option.num_workers, option.max_jobs, option.drop_last, option.in_order, option.timeout)
}


class StatefulTensorDataLoader(dataset: java.StatefulTensorDataset,
                               batch_size: Int,
                               shuffle: Boolean = false,
                               num_workers: Int = 0,
                               max_jobs: Long = 0l,
                               drop_last: Boolean = false,
                               in_order: Boolean = true,
                               timeout: Float = 0)
    extends STDL(dataset, new DLOP())
    with TorchDataLoader with Iterable[TensorExample] {

  val option = TorchTensorDataLoaderOptions(batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, max_jobs = max_jobs, drop_last = drop_last, in_order = in_order, timeout = timeout)

  val nativeDataLoader = new STDL(dataset, option.toNative)

  override def begin(): TensorExampleVectorIterator = nativeDataLoader.begin()

  override def end(): TensorExampleVectorIterator = nativeDataLoader.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = nativeDataLoader.options()


  override def iterator: Iterator[TensorExample] = new Iterator[TensorExample] {

    private var current: TensorExampleIterator =
      nativeDataLoader.begin().asInstanceOf[TensorExampleIterator]

    private val endIterator: TensorExampleIterator =
      nativeDataLoader.end().asInstanceOf[TensorExampleIterator]
    
    override def hasNext: Boolean = !current.equals(endIterator)
    
    override def next(): TensorExample = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}
