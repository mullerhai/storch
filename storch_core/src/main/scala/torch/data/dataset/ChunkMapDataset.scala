package torch.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ExampleOptional,
  ChunkDataReader,
  ExampleStack,
  ExampleVector,
  ExampleVectorOptional,
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
  ChunkMapBatchDataset as CMBD,
  ChunkMapDataset as CMD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}

class ChunkMapDataset(chunkDataset: ChunkSharedBatchDataset)
    extends CMD(chunkDataset)
    with Dataset {


  override def get_batch_example(indices: Long): ExampleOptional = super.get_batch_example(indices)

  override def size(): SizeTOptional = super.size()

  override def dataset(): pytorch.ChunkSharedBatchDataset = super.dataset()

  override def transform(): ExampleStack = super.transform()

  override def get_batch(request: SizeTArrayRef): ExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): ExampleVector = super.get_batch(request: _*)
}




















//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
//
//  override def size(): SizeTOptional = super.size()

//  override def get_batch(request: SizeTArrayRef): ExampleVector = super.get_batch(request)
//
//  override def get_batch(request: Long*): ExampleVector = super.get_batch(request: _*)
//
//  override def size(): SizeTOptional = super.size()
