package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  TensorExampleVectorOptional,
  ChunkTensorDataReader,
  SizeTOptional,
  TensorExampleStack,
  ChunkSharedTensorBatchDataset as CSTBD,
  RandomSampler as RS,
  SequentialSampler as SS
}

class ChunkSharedTensorBatchDataset(chunkTensorDataset: ChunkTensorDataset)
    extends CSTBD(chunkTensorDataset) {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def multiply(): pytorch.ChunkTensorDataset = super.multiply()

  override def access(): pytorch.ChunkTensorDataset = super.access()

  override def reset(): Unit = super.reset()

  override def map(transform: TensorExampleStack): pytorch.ChunkMapTensorDataset =
    super.map(transform)
}
