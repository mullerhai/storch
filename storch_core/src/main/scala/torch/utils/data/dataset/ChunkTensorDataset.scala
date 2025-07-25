package torch.utils.data.dataset

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ChunkDatasetOptions, ExampleVectorOptional, InputArchive, OutputArchive, SizeTOptional, SizeTVectorOptional, T_TensorT_TensorTensor_T_T, T_TensorTensor_T, T_TensorTensor_TOptional, TensorExampleVectorOptional, TensorMapper, TensorVector, TransformerImpl, TransformerOptions, kCircular, kGELU, kReflect, kReplicate, kZeros, ChunkBatchDataset as CBD, ChunkTensorDataReader as CTDR, ChunkTensorDataset as CTD, RandomSampler as RS, SequentialSampler as SS}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.datareader.ChunkTensorDataReader
import torch.utils.data.datareader

class ChunkTensorDataset(
                          chunkReader: datareader.ChunkTensorDataReader,
                          sampler: RS,
                          datasetOptions: ChunkDatasetOptions
) extends CTD(chunkReader, sampler, sampler, datasetOptions) {

  override def get_batch(batch_size: Long): TensorExampleVectorOptional =
    super.get_batch(batch_size)

  override def get_batch(): TensorExampleVectorOptional = super.get_batch()

  override def reset(): Unit = super.reset()

  override def size(): SizeTOptional = super.size()

  override def chunk_sampler(): RS = super.chunk_sampler()

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
