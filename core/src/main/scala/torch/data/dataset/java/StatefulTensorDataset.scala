package torch.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  JavaStatefulTensorDataset as STD,
  TensorExampleVector,
  TensorExample,
  TensorExampleVectorOptional,
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
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.data.datareader.TensorExampleVectorReader
import org.bytedeco.pytorch.AbstractTensor as Tensor

class StatefulTensorDataset(reader: TensorExampleVectorReader) extends STD with Dataset {

  private val ex = new TensorExampleVector(
    new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0))
  )

  def get(index: Long): TensorExample = {
    ex.get(index)
    //                    return super.get(index);
  }

  override def get_batch(size: Long) = new TensorExampleVectorOptional(reader.tensorExampleVec)

  override def reset(): Unit = {
    //            super.reset()
  }

  override def size = new SizeTOptional(reader.tensorExampleVec.size)

}
