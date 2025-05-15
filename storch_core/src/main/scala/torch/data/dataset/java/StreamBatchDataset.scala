package torch.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaStreamBatchDataset as SBD,
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

class StreamBatchDataset(reader: ExampleVectorReader) extends SBD(reader) with Dataset {

  override def get_batch(request: Long): ExampleVector = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
