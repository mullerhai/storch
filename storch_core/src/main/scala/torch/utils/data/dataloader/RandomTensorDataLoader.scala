package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  SizeTVectorOptional,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorExampleVectorIterator,
  TensorMapper,
  TensorVector,
  ChunkBatchDataset as CBD,
  ChunkRandomDataLoader as CRDL,
  JavaRandomTensorDataLoader as RTDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.java.NormalTensorDataset
import torch.utils.data.sampler.RandomSampler
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.dataset.java
import torch.utils.data.sampler

class RandomTensorDataLoader(
                              dataset: java.NormalTensorDataset,
                              sampler: RandomSampler,
                              option: DataLoaderOptions
) extends RTDL(dataset, sampler, option)
    with TorchDataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
