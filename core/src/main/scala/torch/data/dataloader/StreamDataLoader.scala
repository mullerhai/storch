package torch.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaStreamDataLoader as SDL,
  DataLoaderOptions,
  ExampleVectorIterator,
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
import torch.data.dataset.java.StreamDataset
import torch.data.sampler.StreamSampler
import torch.internal.NativeConverters.{fromNative, toNative}

class StreamDataLoader(dataset: StreamDataset, sampler: StreamSampler, option: DataLoaderOptions)
    extends SDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): ExampleVectorIterator = super.begin()

  override def end(): ExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(
    option
  ) // super.options()
}
