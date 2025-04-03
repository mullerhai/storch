package torch.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaSequentialDataLoader as SDL,
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
import torch.data.dataset.java.JavaDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.data.sampler.SequentialSampler

class SequentialDataLoader(
    dataset: JavaDataset,
    sampler: SequentialSampler,
    option: DataLoaderOptions
) extends SDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): ExampleVectorIterator = super.begin()

  override def end(): ExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
