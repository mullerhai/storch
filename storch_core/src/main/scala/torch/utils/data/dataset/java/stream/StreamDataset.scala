package torch
package utils
package data
package dataset
package java
package stream

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ExampleVector,
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
  JavaStreamDataset as JSD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.datareader.ExampleVectorReader
import torch.utils.data.datareader

class StreamDataset(reader: datareader.ExampleVectorReader) extends JSD {

  val ds = new JSD() {
    val exampleVector = reader.exampleVec

    override def get_batch(size: Long): ExampleVector = exampleVector

    override def size = new SizeTOptional(exampleVector.size)
  }
  override def get_batch(request: Long): ExampleVector = ds.get_batch(request) // reader.exampleVec

  override def size(): SizeTOptional = new SizeTOptional(reader.exampleVec.size)

//  override def position(position: Long): SD = super.position(position)
}
