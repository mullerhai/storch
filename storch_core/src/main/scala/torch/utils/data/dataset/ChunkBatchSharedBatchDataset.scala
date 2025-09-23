package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkMapDataset,
  ExampleStack,
  ChunkDataReader,
  ExampleVectorOptional,
  SizeTOptional,
  ChunkBatchSharedBatchDataset as CBSBD,
}

class ChunkBatchSharedBatchDataset(chunkReader: ChunkDataReader) extends CBSBD(chunkReader) {

  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def map(transform: ExampleStack): ChunkMapDataset = super.map(transform)
}


