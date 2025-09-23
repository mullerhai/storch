package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ExampleVectorOptional,
  ChunkTensorDataReader,
  SizeTArrayRef,
  SizeTOptional,
  TensorExampleVector,
  TensorMapper,
  TensorVector,
  ChunkBatchDataset as CBD,
  ChunkMapTensorBatchDataset as CMTBD,
  RandomSampler as RS,
  SequentialSampler as SS
}

class ChunkMapTensorBatchDataset(chunkMapTensorDataset: ChunkMapTensorDataset)
    extends CMTBD(chunkMapTensorDataset) {

  override def get_batch(request: SizeTArrayRef): TensorExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): TensorExampleVector = super.get_batch(request: _*)

  override def size(): SizeTOptional = super.size()
}


//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
