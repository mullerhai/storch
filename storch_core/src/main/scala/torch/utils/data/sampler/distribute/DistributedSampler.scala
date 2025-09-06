package torch
package utils
package data
package sampler
package distribute

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DistributedSampler as DS,
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
  RandomSampler as RS
}
import torch.utils.data.dataset.java.SDataset
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.javacpp.Pointer
class DistributedSampler(epoch: Int) extends DS(epoch.toNative) with Sampler {

  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def next(batch_size: Long): SizeTVectorOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)

}
