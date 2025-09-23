package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ExampleVectorIterator,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  ExampleVector,
  ExampleIterator,
  ChunkBatchDataset as CBD,
  ChunkRandomDataLoader as CRDL,
  JavaRandomDataLoader as RDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.sampler
import org.bytedeco.pytorch.DataLoaderOptions as DLOP
import torch.utils.data.dataset.java.JavaDataset
import torch.utils.data.sampler.RandomSampler
import torch.utils.data.Dataset as DatasetTrait
object RandomDataLoader {

  def apply(
      dataset: JavaDataset | DatasetTrait,
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

class RandomDataLoader(
    dataset: JavaDataset | DatasetTrait,
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

  val nativeDataLoader = new RDL(dataset, sampler, option.toNative)

  override def begin(): ExampleVectorIterator =
    nativeDataLoader.begin() // exampleVectorIterator //dataset.exampleVector.begin()

  override def end(): ExampleVectorIterator =
    nativeDataLoader.end() // exampleVectorIterator //dataset.exampleVector.end()

  override def join(): Unit = nativeDataLoader.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(
    option.toNative
  ) /// super.options()

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

//  val exampleVectorIterator =ExampleVectorIterator(dataset.exampleVector) //.iterator()
//  val exampleVectorIterator =ExampleVectorIterator(dataset.get_batch()) //

//  val dataloader = new RDL(dataset,sampler,option)
