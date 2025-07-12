package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ChunkMapDataset,
  ExampleIterator,
  ExampleStack,
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
//import torch.utils.data.dataset.ChunkMapDataset
import torch.internal.NativeConverters.{fromNative, toNative}
//import torch.utils.data.dataset.ChunkMapDataset //ChunkSharedBatchDataset  //.map(new ExampleStack)

class ChunkRandomDataLoader(dataset: ChunkMapDataset, option: DataLoaderOptions)
    extends CRDL(dataset, option)
    with DataLoader {

  override def begin(): ExampleIterator = super.begin()

  override def end(): ExampleIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(option)

}
