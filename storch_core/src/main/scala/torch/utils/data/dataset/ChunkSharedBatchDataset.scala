package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDataReader,
  ExampleStack,
  ExampleVectorOptional,
  SizeTOptional,
  ChunkSharedBatchDataset as CSBD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import org.bytedeco.pytorch.ChunkMapDataset as CMD
import torch.utils.data.dataset.ChunkMapDataset

class ChunkSharedBatchDataset(chunkDataset: ChunkDataset) extends CSBD(chunkDataset) {

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
