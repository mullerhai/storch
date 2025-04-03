package torch.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaDistributedSequentialTensorDataLoader as DSTDL,
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
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.dataset.java.TensorDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.data.sampler.DistributedSequentialSampler
class DistributedSequentialTensorDataLoader(
    dataset: TensorDataset,
    sampler: DistributedSequentialSampler,
    option: DataLoaderOptions
) extends DSTDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(option)
}
