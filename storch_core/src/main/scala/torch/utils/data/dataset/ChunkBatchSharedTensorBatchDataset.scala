package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkBatchSharedTensorBatchDataset as CBSTBD,
  ChunkTensorDataReader,
  ChunkMapDataset,
  ExampleStack,
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
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.javacpp.Pointer

// ChunkMapTensorDataset
class ChunkBatchSharedTensorBatchDataset(chunkReader: ChunkTensorDataReader)
    extends CBSTBD(chunkReader) {

  override def get_batch(request: Long): _root_.org.bytedeco.pytorch.TensorExampleVectorOptional =
    super.get_batch(request)

  override def size(): _root_.org.bytedeco.pytorch.SizeTOptional = super.size()

  override def map(
      transform: _root_.org.bytedeco.pytorch.TensorExampleStack
  ): _root_.org.bytedeco.pytorch.ChunkMapTensorDataset = super.map(transform)
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
//
//  override def size(): SizeTOptional = super.size()

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
//
//  override def size(): SizeTOptional = super.size()
//
//  override def map(transform: ExampleStack): ChunkMapDataset = super.map(transform)
