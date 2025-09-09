package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDataset as CDS,
  Sampler,
  ChunkDatasetOptions,
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
import torch.internal.NativeConverters.{fromNative, toNative}

class ChunkDataset(
    chunkReader: ChunkDataReader,
    sampler1: RS,
    sampler2: RS,
    datasetOptions: ChunkDatasetOptions
) extends CDS(chunkReader, sampler1, sampler2, datasetOptions) {

  override def get_batch(batch_size: Long): ExampleVectorOptional =
    super.get_batch(datasetOptions.batch_size().get())

  override def get_batch(): ExampleVectorOptional = super.get_batch()

  override def reset(): Unit = super.reset()

  override def size(): SizeTOptional = super.size()

  override def chunk_sampler(): RS = super.chunk_sampler()

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
//
//  override def size(): SizeTOptional = super.size()
