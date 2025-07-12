package torch
package utils
package data
package dataloader

import org.bytedeco.pytorch.{ChunkMapTensorDataset, ChunkSharedTensorBatchDataset, ChunkTensorDataset, TensorExample, TensorExampleIterator, TensorExampleStack, TensorExampleVector}
import torch.utils.data.dataloader.TorchTensorDataLoaderOptions
import java.nio.file.Paths
import scala.collection.Iterator
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import torch.utils.data.dataset.TorchTensorDataset
import torch.utils.data.sampler.RandomSampler
import torch.utils.data.datareader.ChunkTensorDataReader
import torch.utils.data.dataloader.ChunkRandomTensorDataLoader

// 定义一个可迭代的类，用于遍历用户自定义张量数据集
class TorchTensorDataLoader[ParamType <: DType : Default](dataset: TorchTensorDataset[ParamType], options: TorchTensorDataLoaderOptions) extends Iterable[TensorExample] {
  // 转换用户自定义数据集为 TensorExample 序列
  private def convertDatasetToTensorExamples(): Seq[TensorExample] = {
    val tensorExamples = new ArrayBuffer[TensorExample]()
    for (i <- 0 until dataset.length.toInt) {
      val data = dataset.getItem(i)
      // 这里需要根据实际的 Tensor 类型转换为 native 数据
      val tensorExample = new TensorExample(data.native)
      tensorExamples += tensorExample
    }
    tensorExamples.toSeq
  }

  // 创建 TensorExampleVector
  private def createTensorExampleVector(tensorExamples: Seq[TensorExample]): TensorExampleVector = {
    new TensorExampleVector(tensorExamples*)
  }

  // 创建 ChunkTensorDataReader
  private def createChunkTensorDataReader(tensorExampleVector: TensorExampleVector): ChunkTensorDataReader = {
    val reader = new ChunkTensorDataReader()
    reader(tensorExampleVector)
    reader
  }

  // 创建 ChunkTensorDataset
  private def createChunkTensorDataset(reader: ChunkTensorDataReader, tensorExamples: Seq[TensorExample], options: TorchTensorDataLoaderOptions): ChunkTensorDataset = {
    val prefetch_count = 1
    new ChunkTensorDataset(
      reader,
      new RandomSampler(tensorExamples.size),
      new RandomSampler(tensorExamples.size),
      new org.bytedeco.pytorch.ChunkDatasetOptions(prefetch_count, options.batch_size)
    )
  }

  // 创建 ChunkSharedTensorBatchDataset
  private def createChunkSharedTensorBatchDataset(chunkTensorDataset: ChunkTensorDataset): ChunkMapTensorDataset = {
    new ChunkSharedTensorBatchDataset(chunkTensorDataset).map(new TensorExampleStack)
  }

  // 创建 ChunkMapTensorDataset
  //  private def createChunkMapTensorDataset(chunkSharedTensorBatchDataset: ChunkMapTensorBatchDataset): ChunkMapTensorDataset = {
  //    chunkSharedTensorBatchDataset
  //  }

  // 创建 ChunkRandomTensorDataLoader（假设存在对应的类，根据实际情况调整）
  private def createChunkRandomTensorDataLoader(ds: org.bytedeco.pytorch.ChunkMapTensorDataset, options: TorchTensorDataLoaderOptions): ChunkRandomTensorDataLoader = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(options.batch_size)
    loaderOpts.batch_size.put(options.batch_size)
    // 这里需要替换为实际的 ChunkRandomTensorDataLoader 构造函数
    // 假设存在一个名为 ChunkRandomTensorDataLoader 的类
    //    new org.bytedeco.pytorch.ChunkRandomTensorDataLoader(ds, loaderOpts)
    new ChunkRandomTensorDataLoader(ds, loaderOpts)
  }

  // 初始化内部组件
  private val tensorExamples = convertDatasetToTensorExamples()
  private val tensorExampleVector = createTensorExampleVector(tensorExamples)
  private val reader: ChunkTensorDataReader = createChunkTensorDataReader(tensorExampleVector)
  private val chunkTensorDataset: ChunkTensorDataset = createChunkTensorDataset(reader, tensorExamples, options)
  private val chunkSharedTensorBatchDataset: ChunkMapTensorDataset = createChunkSharedTensorBatchDataset(chunkTensorDataset)
  //  private val chunkMapTensorDataset = createChunkMapTensorDataset(chunkSharedTensorBatchDataset)
  private val nativeDataLoader: ChunkRandomTensorDataLoader = createChunkRandomTensorDataLoader(chunkSharedTensorBatchDataset, options)

  override def iterator: Iterator[TensorExample] = new Iterator[TensorExample] {
    private var current: TensorExampleIterator = nativeDataLoader.begin.asInstanceOf[TensorExampleIterator]
    private val endIterator: TensorExampleIterator = nativeDataLoader.end.asInstanceOf[TensorExampleIterator]

    // 检查是否还有下一个元素
    override def hasNext: Boolean = !current.equals(endIterator)

    // 获取下一个元素并移动迭代器
    override def next(): TensorExample = {
      val batch = current.access
      current = current.increment
      batch
    }
  }
}
