package torch.utils.data.datareader

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ExampleOptional,
  ChunkTensorDataReader as CTDR,
  TensorExample,
  TensorExampleVector,
  ExampleStack,
  ExampleVector,
  ExampleVectorOptional,
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
  TransformerImpl,
  TransformerOptions,
  kCircular,
  kGELU,
  kReflect,
  kReplicate,
  kZeros,
  ChunkBatchDataset as CBD,
  ChunkMapBatchDataset as CMBD,
  ChunkMapDataset as CMD,
  RandomSampler as RS,
  SequentialSampler as SS
}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import org.bytedeco.pytorch.AbstractTensor as Tensor
class ChunkTensorDataReader(batch: Int = 32) extends CTDR with DataReader {

  var tensorExampleVec: TensorExampleVector = new TensorExampleVector(
    new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)),
    new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)),
    new TensorExample(Tensor.create(60.0, 20.0, 50.0, 8.0, 232.0))
  )

  override def read_chunk(chunk_index: Long) =
    tensorExampleVec // new TensorExampleVector(new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0)))

  override def chunk_count: Long = 1

  override def reset(): Unit = {}

  def apply(tensorExampleVector: TensorExampleVector): TensorExampleVector = {
    this.tensorExampleVec = tensorExampleVector
    this.tensorExampleVec
  }

  def apply(tensorSeq: Seq[Tensor]): TensorExampleVector = {
    this.tensorExampleVec = new TensorExampleVector(
      tensorSeq.map(x => new TensorExample(x)).toArray: _*
    )
    this.tensorExampleVec
  }

}
