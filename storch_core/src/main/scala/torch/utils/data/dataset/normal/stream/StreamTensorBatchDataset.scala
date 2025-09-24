package torch
package utils
package data
package dataset
package normal
package stream

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  SizeTOptional,
  TensorExampleVector,
  JavaStreamTensorBatchDataset as STBD
}
import torch.utils.data.datareader.TensorExampleVectorReader

class StreamTensorBatchDataset(reader: datareader.TensorExampleVectorReader) extends STBD(reader) {

  override def get_batch(request: Long): TensorExampleVector = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
