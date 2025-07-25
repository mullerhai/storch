package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{DataLoaderOptions, ExampleVectorIterator, ExampleVectorOptional, FullDataLoaderOptions, InputArchive, OutputArchive, SizeTOptional, SizeTVectorOptional, T_TensorT_TensorTensor_T_T, T_TensorTensor_T, T_TensorTensor_TOptional, TensorMapper, TensorVector, TransformerImpl, TransformerOptions, kCircular, kGELU, kReflect, kReplicate, kZeros, ChunkBatchDataset as CBD, ChunkRandomDataLoader as CRDL, JavaDistributedSequentialDataLoader as DSDL, RandomSampler as RS, SequentialSampler as SS}
import torch.utils.data.dataset.java.JavaDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.sampler.DistributedSequentialSampler
import torch.utils.data.dataset.java
import torch.utils.data.sampler

class DistributedSequentialDataLoader(
                                       dataset: java.JavaDataset,
                                       sampler: DistributedSequentialSampler,
                                       option: DataLoaderOptions
) extends DSDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): ExampleVectorIterator = super.begin()

  override def end(): ExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = super.options()
}
