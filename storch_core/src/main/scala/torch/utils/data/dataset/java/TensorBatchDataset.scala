package torch.utils.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ExampleVectorOptional,
  InputArchive,
  OutputArchive,
  SizeTArrayRef,
  SizeTOptional,
  SizeTVectorOptional,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorExampleVector,
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
  JavaTensorBatchDataset as TBD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.datareader.TensorExampleVectorReader
import torch.utils.data.datareader

class TensorBatchDataset(reader: datareader.TensorExampleVectorReader) extends TBD(reader) {

//  override def get_batch(request: Long): TensorExampleVector = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def get_batch(request: SizeTArrayRef): TensorExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): TensorExampleVector = super.get_batch(request: _*)
}
