package torch.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaRandomTensorDataLoader as RTDL,
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
import torch.data.dataset.java.TensorDataset
import torch.data.sampler.RandomSampler
import torch.internal.NativeConverters.{fromNative, toNative}

class RandomTensorDataLoader(
    dataset: TensorDataset,
    sampler: RandomSampler,
    option: DataLoaderOptions
) extends RTDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
