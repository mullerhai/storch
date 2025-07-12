package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkTensorBatchDataset as CTBD,
  ChunkTensorDataReader,
  ExampleVectorOptional,
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
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}

class ChunkTensorBatchDataset(chunkReader: ChunkTensorDataReader)
    extends CTBD(chunkReader) {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
