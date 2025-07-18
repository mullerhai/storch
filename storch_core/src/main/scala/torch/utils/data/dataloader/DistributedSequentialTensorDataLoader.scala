package torch.utils.data.dataloader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{DataLoaderOptions, ExampleVectorOptional, FullDataLoaderOptions, InputArchive, OutputArchive, SizeTOptional, SizeTVectorOptional, T_TensorT_TensorTensor_T_T, T_TensorTensor_T, T_TensorTensor_TOptional, TensorExampleVectorIterator, TensorMapper, TensorVector, JavaDistributedSequentialTensorDataLoader as DSTDL, RandomSampler as RS, SequentialSampler as SS}
import torch.utils.data.dataset.java.TensorDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.sampler.DistributedSequentialSampler
import torch.utils.data.dataset.java
import torch.utils.data.sampler
class DistributedSequentialTensorDataLoader(
                                             dataset: java.TensorDataset,
                                             sampler: DistributedSequentialSampler,
                                             option: DataLoaderOptions
) extends DSTDL(dataset, sampler, option)
    with DataLoader {

  override def begin(): TensorExampleVectorIterator = super.begin()

  override def end(): TensorExampleVectorIterator = super.end()

  override def join(): Unit = super.join()

  override def options(): FullDataLoaderOptions = new FullDataLoaderOptions(option)
}
