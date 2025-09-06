package torch
package utils
package data
package sampler
package stream

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DistributedSampler as DS,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  BatchSizeOptional,
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
  RandomSampler as RS
}
import torch.utils.data.dataset.java.SDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.javacpp.Pointer
import torch.utils.data.dataset.java.SDataset
import torch.utils.data.sampler.BatchSizeSampler
import org.bytedeco.pytorch.StreamSampler as SS
class StreamSampler(epochSize: Long) extends SS(epochSize) with BatchSizeSampler {

  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def reset(): Unit = super.reset()

  override def next(batch_size: Long): BatchSizeOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}
