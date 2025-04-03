package torch.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDataReader,
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
  TransformerImpl,
  TransformerOptions,
  kCircular,
  kGELU,
  kReflect,
  kReplicate,
  kZeros,
  ChunkBatchDataset as CBD,
  ChunkSharedBatchDataset as CSBD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.pytorch.ChunkMapDataset as CMD
import torch.data.dataset.ChunkMapDataset
class ChunkSharedBatchDataset(chunkDataset: ChunkDataset) extends CSBD(chunkDataset) with Dataset {

  var native: CMD = null


  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def map(transform: ExampleStack): CMD = {

    native = super.map(transform)
//    native.transform()
//    native.asInstanceOf[ChunkMapDataset]
//    new ChunkMapDataset()
    native

  }
}
