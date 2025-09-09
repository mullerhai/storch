package torch
package utils
package data

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Example,
  ExampleVector,
  InputArchive,
  OutputArchive,
  SizeTArrayRef,
  SizeTOptional,
  SizeTVectorOptional,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorMapper,
  TensorVector,
  ChunkBatchDataset as CBD,
  JavaDataset as SD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.internal.NativeConverters.{fromNative, toNative}

trait NormalDataset extends SD {

//  override def get_batch(request: Long): ExampleVector = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def position(position: Long): SD = super.position(position)

  override def getPointer(i: Long): SD = super.getPointer(i)

  override def get(index: Long): Example = super.get(index)

  override def get_batch(indices: SizeTArrayRef): ExampleVector = super.get_batch(indices)

  override def get_batch(request: Long*): ExampleVector = super.get_batch(request: _*)
}
