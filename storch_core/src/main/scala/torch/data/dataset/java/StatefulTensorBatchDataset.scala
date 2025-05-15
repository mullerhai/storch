package torch.data.dataset.java

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
  TransformerImpl,
  TransformerOptions,
  kCircular,
  kGELU,
  kReflect,
  kReplicate,
  kZeros,
  ChunkBatchDataset as CBD,
  JavaStatefulTensorBatchDataset as STBD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.datareader.TensorExampleVectorReader
import torch.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}

class StatefulTensorBatchDataset(reader: TensorExampleVectorReader)
    extends STBD(reader)
    with Dataset {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
