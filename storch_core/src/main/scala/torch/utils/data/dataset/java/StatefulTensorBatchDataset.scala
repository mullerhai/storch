package torch.utils.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  InputArchive,
  OutputArchive,
  SizeTOptional,
  SizeTVectorOptional,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorExampleVectorOptional,
  TensorMapper,
  TensorVector,
  ChunkBatchDataset as CBD,
  JavaStatefulTensorBatchDataset as STBD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.datareader.TensorExampleVectorReader
import torch.utils.data.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.datareader

class StatefulTensorBatchDataset(reader: datareader.TensorExampleVectorReader)
    extends STBD(reader) {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
