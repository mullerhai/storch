package torch
package utils
package data
package dataloader
package distribute

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
  JavaDistributedRandomTensorDataLoader as DRTDL,
  SequentialSampler as SS
}
import torch.utils.data.dataset.java.NormalTensorDataset
import torch.utils.data.sampler.distribute.DistributedRandomSampler
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.dataset.java
import torch.utils.data.sampler

class DistributedRandomTensorDataLoader(
                                         dataset: NormalTensorDataset,
                                         sampler: DistributedRandomSampler,
                                         option: DataLoaderOptions
) extends DRTDL(dataset, sampler, option)
    with TorchDataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
