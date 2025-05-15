package torch.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaStreamTensorDataset as STD,
  TensorExampleVector,
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

import torch.data.datareader.TensorExampleVectorReader
class StreamTensorDataset(reader: TensorExampleVectorReader) extends STD with Dataset {

  override def get_batch(request: Long): TensorExampleVector = reader.tensorExampleVec

  override def size(): SizeTOptional = new SizeTOptional(reader.tensorExampleVec.size)
}
