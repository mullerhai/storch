package torch.utils.data.sampler

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Sampler as SM,
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
  RandomSampler as RS
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.Dataset
trait Sampler extends SM {

//  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)
//
//  override def next(batch_size: Long): SizeTVectorOptional = super.next(batch_size)
//
//  override def save(archive: OutputArchive): Unit = super.save(archive)
//
//  override def load(archive: InputArchive): Unit = super.load(archive)
}
