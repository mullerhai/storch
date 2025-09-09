package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ChunkMapTensorDataset,
  ChunkSharedTensorBatchDataset,
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
  TensorExampleIterator,
  TensorExampleStack,
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
  ChunkRandomTensorDataLoader as CRTDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
//import torch.utils.data.dataset.ChunkMapTensorDataset
import torch.internal.NativeConverters.{fromNative, toNative}
//import torch.utils.data.dataset.ChunkSharedTensorBatchDataset .map(new TensorExampleStack)
class ChunkRandomTensorDataLoader(dataset: ChunkMapTensorDataset, option: DataLoaderOptions)
    extends CRTDL(dataset, option)
    with TorchDataLoader {

  override def begin(): TensorExampleIterator = super.begin()

  override def end(): TensorExampleIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}

//  override def begin(): ExampleIterator = super.begin()
//
//  override def end(): ExampleIterator = super.end()
//
//  override def join(): Unit = super.join()
//
//  override def options(): FullDataLoaderOptions = super.options()
