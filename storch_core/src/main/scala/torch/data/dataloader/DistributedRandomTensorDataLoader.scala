package torch.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaDistributedRandomTensorDataLoader as DRTDL,
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
  SequentialSampler as SS
}
import torch.data.dataset.java.TensorDataset
import torch.data.sampler.DistributedRandomSampler
import torch.internal.NativeConverters.{fromNative, toNative}

class DistributedRandomTensorDataLoader(
    dataset: TensorDataset,
    sampler: DistributedRandomSampler,
    option: DataLoaderOptions
) extends DRTDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
