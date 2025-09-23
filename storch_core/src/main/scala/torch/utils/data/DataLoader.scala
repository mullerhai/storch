package torch
package utils
package data

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.Tensor as TensorNative
import torch.utils.data.dataloader.{TorchDataLoaderOptions, SequentialDataLoader}
import torch.utils.data.datareader.ChunkDataReader
import org.bytedeco.javacpp.chrono.Milliseconds
import torch.utils.data.sampler.{RandomSampler as TorchSampler, SequentialSampler}
import torch.{DType, Default}
//import java.nio.file.Paths
import scala.collection.Iterator
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import torch.utils.data.Dataset as DatasetTrait
import torch.utils.data.dataset.java.JavaDataset
import torch.{Tensor, *}
import torch.internal.NativeConverters.{fromNative, toNative}

class DataLoader[ParamType <: DType: Default](
    dataset: DatasetTrait | JavaDataset,
    batch_size: Int,
    shuffle: Boolean = false,
    num_workers: Int = 0,
    max_jobs: Long = 0L,
    drop_last: Boolean = false,
    in_order: Boolean = true,
    sampler: TorchSampler,
    batch_sampler: TorchSampler,
    timeout: Float = 0,
    pin_memory: Boolean = false,
    prefetch_factor: Option[Int] = None,
    persistent_workers: Boolean = false,
    pin_memory_device: String = ""
) extends Iterable[(Tensor[ParamType], Tensor[ParamType])] {

  private val options = TorchDataLoaderOptions(
    batch_size = batch_size,
    shuffle = shuffle,
    sampler = sampler,
    batch_sampler = batch_sampler,
    num_workers = num_workers,
    max_jobs = max_jobs,
    pin_memory = pin_memory,
    drop_last = drop_last,
    in_order = in_order,
    timeout = timeout,
    prefetch_factor = prefetch_factor,
    persistent_workers = persistent_workers
  )

  private def convertDatasetToExamples(): Seq[Example] = {
    val examples = new ArrayBuffer[Example]()
    if (dataset.isInstanceOf[DatasetTrait]) {
      val datasetTrait = dataset.asInstanceOf[DatasetTrait]
      for (i <- 0 until datasetTrait.length.toInt) {
        val (data, target) = datasetTrait.getItem(i)
        val example = new Example(data.native, target.native)
        examples += example
      }
    } else {
      val javaDataset = dataset.asInstanceOf[JavaDataset]
      for (i <- 0 until javaDataset.length.toInt) {
        val example = javaDataset.get(i)
        examples += example
      }
    }
    examples.toSeq
  }

  def exampleVectorToExample(exVec: ExampleVector): Example = {
    val example = new Example(exVec.get(0).data(), exVec.get(0).target())
    example
  }

  private def createChunkDataReader(examples: Seq[Example]): ChunkDataReader = {
    val reader = new ChunkDataReader()
    val exampleVector = new org.bytedeco.pytorch.ExampleVector(examples*)
    reader(exampleVector)
    reader
  }

  private def createChunkDataset(
      reader: ChunkDataReader,
      examples: Seq[Example],
      options: TorchDataLoaderOptions
  ): ChunkDataset = {

    if (options.shuffle) {
      // for random sampler
      new ChunkDataset(
        reader,
        sampler,
        batch_sampler,
        new ChunkDatasetOptions(prefetch_factor.getOrElse(2), options.batch_size.toLong)
      )
    } else {
      // for sequential sampler
      new ChunkDataset(
        reader,
        sampler,
        batch_sampler,
        new ChunkDatasetOptions(prefetch_factor.getOrElse(2), options.batch_size.toLong)
      )
    }

  }

  private def createChunkSharedBatchDataset(chunkDataset: ChunkDataset): ChunkMapDataset = {
    new ChunkSharedBatchDataset(chunkDataset).map(new ExampleStack)
  }

  private def createChunkRandomDataLoader(
      ds: ChunkMapDataset,
      options: TorchDataLoaderOptions
  ): ChunkRandomDataLoader = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(options.batch_size)
    loaderOpts.batch_size.put(options.batch_size)
    loaderOpts.timeout().put(new Milliseconds(options.timeout.toLong))
    loaderOpts.drop_last().put(options.drop_last)
    loaderOpts.enforce_ordering().put(options.in_order)
    loaderOpts.workers().put(options.num_workers)
    loaderOpts.max_jobs().put(options.max_jobs)
    new ChunkRandomDataLoader(ds, loaderOpts)
  }

  private def createChunkSequentialDataLoader(
      ds: JavaDataset,
      sampler: SequentialSampler,
      options: TorchDataLoaderOptions
  ): SequentialDataLoader = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(options.batch_size)
    loaderOpts.batch_size.put(options.batch_size)
    loaderOpts.timeout().put(new Milliseconds(options.timeout.toLong))
    loaderOpts.drop_last().put(options.drop_last)
    loaderOpts.enforce_ordering().put(options.in_order)
    loaderOpts.workers().put(options.num_workers)
    loaderOpts.max_jobs().put(options.max_jobs)
    SequentialDataLoader(ds, sampler, options)
  }

  private val examples = convertDatasetToExamples()
  private val reader = createChunkDataReader(examples)

  private val nativeDataset: ChunkDataset = createChunkDataset(reader, examples, options)
  private val sharedBatchDataset = createChunkSharedBatchDataset(nativeDataset)
  private val nativeDataLoader: ChunkRandomDataLoader =
    createChunkRandomDataLoader(sharedBatchDataset, options)

  private def exampleToTuple(example: Example): (Tensor[ParamType], Tensor[ParamType]) = {
    val feature = fromNative(example.data()).to(dtype = implicitly[Default[ParamType]].dtype)
    val target = fromNative(example.target()).to(dtype = implicitly[Default[ParamType]].dtype)
    (feature, target)
  }

  override def iterator: Iterator[(Tensor[ParamType], Tensor[ParamType])] =
    new Iterator[(Tensor[ParamType], Tensor[ParamType])] {

      private var current: ExampleIterator = nativeDataLoader.begin

      private val endIterator: ExampleIterator = nativeDataLoader.end

      override def hasNext: Boolean = !current.equals(endIterator)

      override def next(): (Tensor[ParamType], Tensor[ParamType]) = {
        val batch = current.access
        current = current.increment
        exampleToTuple(batch)
      }
    }
}

//    val prefetch_count = 1
//    new ChunkDataset(
//      reader,
//      new RandomSampler(examples.size),
//      new RandomSampler(examples.size),
//      new ChunkDatasetOptions(prefetch_count, options.batch_size.toLong)
//    )
//    dataset: Dataset[ParamType], batchSize: Int, shuffle: Boolean, sampler: Sampler, batchSampler: Sampler[List],
//    numWorkers: Int = 0, pinMemory: Boolean = false, dropLast: Boolean = false, timeout: Float = 0,
//    multiprocessingContext: ProcessContext = null, generator: Generator ,

//import torch.utils.data.dataloader.ChunkRandomDataLoader
//    train_loader = DataLoader(
//        train_ds,
//        batch_size=args.batch_size,
//        pin_memory=True,
//        drop_last=False,
//        shuffle=False,
//        num_workers=args.num_workers,
//        sampler=train_sampler
//    )
// dataset: Dataset[_T_co],
//        batch_size: Optional[int] = 1,
//        shuffle: Optional[bool] = None,
//        sampler: Union[Sampler, Iterable, None] = None,
//        batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
//        num_workers: int = 0,
//        collate_fn: Optional[_collate_fn_t] = None,
//        pin_memory: bool = False,
//        drop_last: bool = False,
//        timeout: float = 0,
//        worker_init_fn: Optional[_worker_init_fn_t] = None,
//        multiprocessing_context=None,
//        generator=None,
//        *,
//        prefetch_factor: Optional[int] = None,
//        persistent_workers: bool = False,
//        pin_memory_device: str = "",
//        in_order: bool = True,
// 定义一个可迭代的类，用于遍历用户自定义数据集
// batchSize -> batch_size ,
// shuffle inOrder in_order-> enforce_ordering ,
// numWorkers -> num_workers ->workers,
// max_jobs
// pinMemory -> pin_memory ,
// dropLast -> drop_last ,
// timeout -> timeout ,
//
