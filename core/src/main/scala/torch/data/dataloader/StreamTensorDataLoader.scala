package torch.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaStreamTensorDataLoader as STDL,
  DataLoaderOptions,
  TensorExampleVectorIterator,
  ExampleVectorOptional,
  FullDataLoaderOptions,
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
  ChunkRandomDataLoader as CRDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.dataset.java.StreamTensorDataset
import torch.data.sampler.StreamSampler
import torch.internal.NativeConverters.{fromNative, toNative}

class StreamTensorDataLoader(
    dataset: StreamTensorDataset,
    sampler: StreamSampler,
    option: DataLoaderOptions
) extends STDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
