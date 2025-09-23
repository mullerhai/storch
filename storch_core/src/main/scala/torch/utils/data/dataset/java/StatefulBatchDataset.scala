package torch.utils.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ExampleVectorOptional, SizeTOptional, JavaStatefulBatchDataset as SBD}
import torch.utils.data.datareader.ExampleVectorReader
import torch.utils.data.datareader

class StatefulBatchDataset(reader: datareader.ExampleVectorReader) extends SBD(reader) {

  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
