package torch.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Example,
  ExampleVector,
  ChunkMapDataset,
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
import torch.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.data.datareader.ExampleVectorReader

class JavaDataset(exampleVectorReader: ExampleVectorReader) extends JD with Dataset {

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

    // ds.get(index) //exampleVector.get(index)
//   println(s"example shape  ${example.data().print()}")

//    println(s"oldExample shape ${oldExample.data().print()} ")
    val flag = example.equals(oldExample)
//    println(s"flag ${flag} ")
    if (flag) oldExample else example
//    oldExample
  } // ds.get(index) //exampleVector.get(index)

  override def size(): SizeTOptional = ds.size() // new SizeTOptional(exampleVector.size)

  override def get_batch(indices: SizeTArrayRef): ExampleVector =
    super.get_batch(indices) // ds.get_batch(indices) // exampleVector
}
