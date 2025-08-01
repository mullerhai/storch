package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DataLoaderOptions,
  ExampleVectorIterator,
  ExampleVectorOptional,
  FullDataLoaderOptions,
  InputArchive,
  JavaDataset,
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
  JavaRandomDataLoader as RDL,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.sampler
//import torch.utils.data.dataset.java.JavaDataset
import torch.utils.data.sampler.RandomSampler
import torch.internal.NativeConverters.{fromNative, toNative}

class RandomDataLoader(dataset: JavaDataset, sampler: RandomSampler, option: DataLoaderOptions)
    extends RDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): ExampleVectorIterator =
    super.begin() // exampleVectorIterator //dataset.exampleVector.begin()

  override def end(): ExampleVectorIterator =
    super.end() // exampleVectorIterator //dataset.exampleVector.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(
    option
  ) /// super.options()

}

//  val exampleVectorIterator =ExampleVectorIterator(dataset.exampleVector) //.iterator()
//  val exampleVectorIterator =ExampleVectorIterator(dataset.get_batch()) //

//  val dataloader = new RDL(dataset,sampler,option)
