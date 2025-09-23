package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkTensorDataReader,
  ExampleVectorOptional,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  TensorExampleVectorOptional,
  ChunkStatefulTensorDataset as CSTD,
  RandomSampler as RS,
  SequentialSampler as SS
}

class ChunkStatefulTensorDataset(chunkReader: ChunkTensorDataReader) extends CSTD(chunkReader) {

  override def reset(): Unit = super.reset()

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
