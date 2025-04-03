package torch.data.sampler

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
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
import torch.data.dataset.Dataset

trait BatchSizeSampler extends BSS
