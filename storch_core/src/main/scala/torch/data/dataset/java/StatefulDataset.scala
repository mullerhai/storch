package torch.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaStatefulDataset as SD,
  ExampleVectorOptional,
  ExampleVector,
  Example,
  InputArchive,
  OutputArchive,
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
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.pytorch.AbstractTensor as Tensor
import torch.data.datareader.ExampleVectorReader

class StatefulDataset(reader: ExampleVectorReader) extends SD(reader) with Dataset {

  private val ex1 = new Example(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(200.0))
  private val ex2 = new Example(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(400.0))
  private val ex3 = new Example(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(500.0))
  val exampleVector = new ExampleVector(ex1, ex2, ex3)

  override def get_batch(size: Long) = new ExampleVectorOptional(reader.exampleVec)

  override def reset(): Unit = {
    //                    super.reset();
  }

  override def size = new SizeTOptional(reader.exampleVec.size)
}
