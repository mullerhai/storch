package torch.utils.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  SizeTOptional,
  TensorExample,
  TensorExampleVector,
  TensorExampleVectorOptional,
  JavaStatefulTensorDataset as STD
}
import torch.utils.data.datareader.TensorExampleVectorReader
import org.bytedeco.pytorch.AbstractTensor as Tensor
import torch.utils.data.datareader

class StatefulTensorDataset(reader: datareader.TensorExampleVectorReader) extends STD {

  override def get_batch(size: Long) = new TensorExampleVectorOptional(reader.tensorExampleVec)

  override def size = new SizeTOptional(reader.tensorExampleVec.size)

  override def reset(): Unit = {
//    reader.tensorExampleVec =  new TensorExampleVector()
  }

  private val exTest = new TensorExampleVector(
    new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0))
  )

  def getTest(index: Long): TensorExample = {
    exTest.get(index)
  }

}
