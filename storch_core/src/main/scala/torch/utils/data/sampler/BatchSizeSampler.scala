package torch.utils.data.sampler

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  BatchSizeOptional,
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
  BatchSizeSampler as BSS,
  RandomSampler as RS
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.dataset.Dataset

trait BatchSizeSampler extends BSS {
  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def next(batch_size: Long): BatchSizeOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}
