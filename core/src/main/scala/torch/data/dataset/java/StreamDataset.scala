package torch.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaStreamDataset as JSD,
  ExampleVector,
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
import torch.data.datareader.ExampleVectorReader

class StreamDataset(reader: ExampleVectorReader) extends JSD with Dataset {

  val ds = new JSD() {
    val exampleVector = reader.exampleVec

    override def get_batch(size: Long): ExampleVector = exampleVector

    override def size = new SizeTOptional(exampleVector.size)
  }
  override def get_batch(request: Long): ExampleVector = ds.get_batch(request) // reader.exampleVec

  override def size(): SizeTOptional = new SizeTOptional(reader.exampleVec.size)

//  override def position(position: Long): SD = super.position(position)
}
