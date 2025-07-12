package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkMapDataset,
  ExampleStack,
  ChunkDataReader,
  ExampleVectorOptional,
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
  ChunkBatchSharedBatchDataset as CBSBD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.dataset.Dataset

//ChunkMapDataset
class ChunkBatchSharedBatchDataset(chunkReader: ChunkDataReader)
    extends CBSBD(chunkReader) {

  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def map(transform: ExampleStack): ChunkMapDataset = super.map(transform)
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
//
//  override def size(): SizeTOptional = super.size()
