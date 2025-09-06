package torch
package utils
package data
package dataset
package java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkMapDataset,
  Example,
  ExampleVector,
  InputArchive,
  OutputArchive,
  SizeTArrayRef,
  SizeTOptional,
  SizeTVectorOptional,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorMapper,
  TensorVector,
  TransformerImpl,
  TransformerOptions,
  kCircular,
  kGELU,
  kReflect,
  kReplicate,
  kZeros,
  ChunkBatchDataset as CBD,
  JavaBatchDataset as BD,
  JavaDataset as JD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.datareader.ExampleVectorReader
import torch.utils.data.datareader
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class JavaDataset(exampleVectorReader: datareader.ExampleVectorReader) extends JD {

  val exampleVector: ExampleVector = exampleVectorReader.exampleVec // .read_chunk(0)
//    {
//    val batch = 32 //exampleVectorReader.batch
//    val exampleVector = new ExampleVector()
//    for (i <- 0 until batch) {
//       exampleVector.put( exampleVectorReader.read_chunk(i.toLong).get*)
//    }
//    exampleVector
//  }

  //     val exampleVector = new ExampleVector(exampleSeq.toArray:_*)
  //    override def get(index: Long): Example = exampleVector.get(index)
  //
  //    override def size = new SizeTOptional(exampleVector.size)
  //

  val ds = new JD() {
    val exampleVector = exampleVectorReader.exampleVec // new ExampleVector(exampleSeq.toArray:_*)
    override def get(index: Long): Example = exampleVector.get(index)

    override def size = new SizeTOptional(exampleVector.size)

  }
//  override def get(index: Long): Example = super.get(index) // 失败
  override def get(index: Long): Example = {
    val example = ds.get(index)
    val oldExample = exampleVector.get(index)
//   println(s"example shape  ${example.data().print()} oldExample shape ${oldExample.data().print()}")
    val flag = example.equals(oldExample)
    if (flag) oldExample else example

  } // ds.get(index) //exampleVector.get(index)

  override def size(): SizeTOptional = ds.size() // new SizeTOptional(exampleVector.size)

  override def get_batch(indices: SizeTArrayRef): ExampleVector =
    super.get_batch(indices) // ds.get_batch(indices) // exampleVector

  def randomSplit(lengths: Seq[Int], seed: Long = System.currentTimeMillis()): Array[JavaDataset] = {
    require(lengths.sum == size().get(), "切分长度总和必须等于数据集大小")

    // 创建索引序列并随机打乱
    val indices = ArrayBuffer.range(0, size().get().toInt)
    val random = new Random(seed)
    for (i <- indices.length - 1 to 1 by -1) {
      val j = random.nextInt(i + 1)
      val temp = indices(i)
      indices(i) = indices(j)
      indices(j) = temp
    }

    // 分割索引并创建子数据集
    val splitDatasets = ArrayBuffer[JavaDataset]()
    var current = 0

    for (length <- lengths) {
      val end = current + length
      val subsetIndices = indices.slice(current, end)
      // 创建子数据集读取器
      val subsetReader = new ExampleVectorReader {
        override val exampleVec: ExampleVector = {
          val vec = new ExampleVector()
          subsetIndices.foreach(idx => vec.put(ds.get(idx.toLong)))
          vec
        }
      }

      splitDatasets += new JavaDataset(subsetReader)
      current = end
    }

    splitDatasets.toArray
  }

  /**
   * 按比例随机切分数据集
   *
   * @param ratios 每个子数据集的比例
   * @param seed   随机种子
   * @return 切分后的子数据集数组
   */
  def randomSplitByRatio(ratios: Seq[Double], seed: Long = System.currentTimeMillis()): Array[JavaDataset] = {
    require(math.abs(ratios.sum - 1.0) < 1e-6, "比例总和必须接近1.0")

    val totalSize = size().get().toInt
    val lengths = ratios.map(ratio => (ratio * totalSize).toInt)

    // 处理四舍五入误差
    val remaining = totalSize - lengths.sum
    if (remaining > 0) {
      lengths.updated(lengths.length - 1, lengths.last + remaining)
    }

    randomSplit(lengths, seed)
  }
}