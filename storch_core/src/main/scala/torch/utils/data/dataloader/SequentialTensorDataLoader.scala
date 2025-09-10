package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ExampleIterator,
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
  JavaSequentialTensorDataLoader as STDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.java.NormalTensorDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.sampler.SequentialSampler
import torch.utils.data.dataset.java
import torch.utils.data.sampler

class SequentialTensorDataLoader(
    dataset: java.NormalTensorDataset,
    sampler: SequentialSampler,
    option: DataLoaderOptions
) extends STDL(dataset, sampler, option)
    with TorchDataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
