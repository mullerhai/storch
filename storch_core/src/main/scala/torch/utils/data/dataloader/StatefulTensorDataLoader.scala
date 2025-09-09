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
  JavaStatefulTensorDataLoader as STDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.java.StatefulTensorDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.dataset.java

class StatefulTensorDataLoader(dataset: java.StatefulTensorDataset, option: DataLoaderOptions)
    extends STDL(dataset, option)
    with TorchDataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
