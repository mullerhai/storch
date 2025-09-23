package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkTensorBatchDataset as CTBD,
  ChunkTensorDataReader,
  SizeTOptional,
  TensorExampleVectorOptional
}

class ChunkTensorBatchDataset(chunkReader: ChunkTensorDataReader) extends CTBD(chunkReader) {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
