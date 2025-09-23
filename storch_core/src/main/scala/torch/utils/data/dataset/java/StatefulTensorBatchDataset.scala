package torch.utils.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  SizeTOptional,
  TensorExampleVectorOptional,
  JavaStatefulTensorBatchDataset as STBD,
}
import torch.utils.data.datareader.TensorExampleVectorReader

import torch.utils.data.datareader

class StatefulTensorBatchDataset(reader: datareader.TensorExampleVectorReader)
    extends STBD(reader) {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
