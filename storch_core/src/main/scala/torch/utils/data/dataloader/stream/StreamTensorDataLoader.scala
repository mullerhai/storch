package torch
package utils
package data
package dataloader
package stream

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
  JavaStreamTensorDataLoader as STDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.java.stream.StreamTensorDataset
import torch.utils.data.sampler.stream.StreamSampler
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.dataset.java
import torch.utils.data.sampler

class StreamTensorDataLoader(
    dataset: StreamTensorDataset,
    sampler: StreamSampler,
    option: DataLoaderOptions
) extends STDL(dataset, sampler, option)
    with TorchDataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
