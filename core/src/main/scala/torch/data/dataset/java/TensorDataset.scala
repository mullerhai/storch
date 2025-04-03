package torch.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  InputArchive,
  OutputArchive,
  SizeTArrayRef,
  SizeTOptional,
  SizeTVectorOptional,
  T_TensorT_TensorTensor_T_T,
  T_TensorTensor_T,
  T_TensorTensor_TOptional,
  TensorExample,
  TensorExampleVector,
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
  JavaTensorDataset as TD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.data.datareader.TensorExampleVectorReader

class TensorDataset(reader: TensorExampleVectorReader) extends TD with Dataset {

  var tensorExampleVec: TensorExampleVector = reader.read_chunk(0)
//  override def get_batch(request: Long): TensorExampleVector = super.get_batch(request)

  val ds = new TD() {
    val tex = reader.read_chunk(
      0
    ) // new TensorExampleVector(new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0)))

    override def get(index: Long): TensorExample = {
      tex.get(index)
      //                    return super.get(index);
    }

    override def get_batch(indices: SizeTArrayRef): TensorExampleVector =
      tex // .get_batch(indices) // ds.get_batch(indices) // exampleVector
    override def size = new SizeTOptional(tex.size)
  }
  override def get(index: Long): TensorExample =
    ds.get(index) // ds.get(index) //exampleVector.get(index) {
//    tensorExampleVec.get(index)
//    //                    return super.get(index);
//  }

  override def size = new SizeTOptional(tensorExampleVec.size)

  override def position(position: Long): TD = super.position(position)

  override def getPointer(i: Long): TD = super.getPointer(i)

  override def get_batch(indices: SizeTArrayRef): TensorExampleVector =
    ds.get_batch(indices) // ds.get_batch(indices) // exampleVector
}
