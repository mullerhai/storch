package torch
package utils
package data
package dataloader

import java.nio.file.Paths
import scala.collection.Iterator
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import org.bytedeco.pytorch.{
  ChunkDataset,
  ChunkDatasetOptions,
  ChunkMapDataset,
  ChunkRandomDataLoader,
  ChunkSharedBatchDataset,
  Example,
  ExampleIterator,
  ExampleStack,
  ExampleVector
}
import torch.utils.data.dataset.Dataset
import torch.utils.data.sampler.RandomSampler
import torch.utils.data.datareader.ChunkDataReader
//import torch.utils.data.dataloader.ChunkRandomDataLoader

// 定义一个可迭代的类，用于遍历用户自定义数据集
class TorchDataLoader[ParamType <: DType: Default](
    dataset: Dataset[ParamType],
    options: TorchDataLoaderOptions
) extends Iterable[Example] {
  // 转换用户自定义数据集为 Example 序列
  private def convertDatasetToExamples(): Seq[Example] = {
    val examples = new ArrayBuffer[Example]()
    for (i <- 0 until dataset.length.toInt) {
      val (data, target) = dataset.getItem(i)
      // 这里需要根据实际的 Tensor 类型转换为 native 数据
      val example = new Example(data.native, target.native)
      examples += example
    }
    examples.toSeq
  }

  def exampleVectorToExample(exVec: ExampleVector): Example = {
    val example = new Example(exVec.get(0).data(), exVec.get(0).target())
    example
  }
  // 创建 ChunkDataReader
  private def createChunkDataReader(examples: Seq[Example]): ChunkDataReader = {
    val reader = new ChunkDataReader()
    val exampleVector = new org.bytedeco.pytorch.ExampleVector(examples*)
    reader(exampleVector)
    reader
  }

  // 创建 ChunkDataset
  private def createChunkDataset(
      reader: ChunkDataReader,
      examples: Seq[Example],
      options: TorchDataLoaderOptions
  ): ChunkDataset = {

    val prefetch_count = 1
    new ChunkDataset(
      reader,
      new RandomSampler(examples.size),
      new RandomSampler(examples.size),
      new ChunkDatasetOptions(prefetch_count, options.batch_size.toLong)
    )
  }

  // 创建 ChunkSharedBatchDataset
  private def createChunkSharedBatchDataset(chunkDataset: ChunkDataset): ChunkMapDataset = {
    new ChunkSharedBatchDataset(chunkDataset).map(new ExampleStack)
  }

  // 创建 ChunkRandomDataLoader
  private def createChunkRandomDataLoader(
      ds: ChunkMapDataset,
      options: TorchDataLoaderOptions
  ): ChunkRandomDataLoader = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(options.batch_size)
    loaderOpts.batch_size.put(options.batch_size)
    //    loaderOpts.timeout().put(new Milliseconds(options.timeout.toLong))
    loaderOpts.drop_last().put(options.drop_last)
    loaderOpts.enforce_ordering().put(!options.shuffle)
    loaderOpts.workers().put(options.num_workers)
    loaderOpts.max_jobs().put(4)
    new ChunkRandomDataLoader(ds, loaderOpts)
  }

  // 初始化内部组件
  private val examples = convertDatasetToExamples()
  private val reader = createChunkDataReader(examples)
  private val nativeDataset: ChunkDataset = createChunkDataset(reader, examples, options)
  private val sharedBatchDataset = createChunkSharedBatchDataset(nativeDataset)
  private val nativeDataLoader: ChunkRandomDataLoader =
    createChunkRandomDataLoader(sharedBatchDataset, options)

  override def iterator: Iterator[Example] = new Iterator[Example] {
    private var current: ExampleIterator = nativeDataLoader.begin
    private val endIterator: ExampleIterator = nativeDataLoader.end

    // 检查是否还有下一个元素
    override def hasNext: Boolean = !current.equals(endIterator)

    // 获取下一个元素并移动迭代器
    override def next(): Example = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}
