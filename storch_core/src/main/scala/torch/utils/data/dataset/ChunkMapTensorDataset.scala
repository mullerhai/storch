package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ExampleVectorOptional,
  ChunkTensorDataReader,
  InputArchive,
  OutputArchive,
  SizeTArrayRef,
  SizeTOptional,
  SizeTVectorOptional,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorExampleOptional,
  TensorExampleStack,
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
  ChunkMapTensorDataset as CMTD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}

class ChunkMapTensorDataset(chunkDataset: ChunkSharedTensorBatchDataset)
    extends CMTD(chunkDataset) {

  override def get_batch_example(indices: Long): TensorExampleOptional =
    super.get_batch_example(indices)

  override def size(): SizeTOptional = super.size()

  override def dataset(): pytorch.ChunkSharedTensorBatchDataset = super.dataset()

  override def transform(): TensorExampleStack = super.transform()

  override def get_batch(request: SizeTArrayRef): TensorExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): TensorExampleVector = super.get_batch(request: _*)
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
