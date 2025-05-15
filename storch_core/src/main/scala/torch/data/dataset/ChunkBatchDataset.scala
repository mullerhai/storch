package torch.data.dataset

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
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
  ChunkBatchDataset as CBD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
// with Dataset
class ChunkBatchDataset(chunkReader: ChunkDataReader) extends CBD(chunkReader) with Dataset {

  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def offsetAddress[P <: Pointer](i: Long): P = super.offsetAddress(i)
}

object ChunkBatchDataset {}
