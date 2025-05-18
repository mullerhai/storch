package torch.data.datareader

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.{
  AbstractTensor as Tensors,
  ExampleOptional,
  Example,
  Tensor,
  ExampleVector,
  ExampleVectorOptional,
  InputArchive,
  OutputArchive,
  SizeTArrayRef,
  SizeTOptional,
  SizeTVectorOptional,
  ChunkBatchDataset as CBD,
  ChunkMapBatchDataset as CMBD,
  ChunkMapDataset as CMD,
  RandomSampler as RS,
  SequentialSampler as SS
}

final class ExampleVectorReader(batch: Int = 32) extends Pointer with DataReader {

  var exampleVec: ExampleVector =
    new ExampleVector() // new Example(Tensors.create(10.0, 20.0, 50.0, 80.0, 100.0), Tensors.create(200.0)), new Example(Tensors.create(15.0, 30.0, 50.0, 80.0, 300.0), Tensors.create(400.0)), new Example(Tensors.create(20.0, 20.0, 50.0, 80.0, 100.0), Tensors.create(500.0)), new Example(Tensors.create(35.0, 30.0, 50.0, 80.0, 300.0), Tensors.create(600.0)), new Example(Tensors.create(40.0, 20.0, 50.0, 80.0, 100.0), Tensors.create(700.0)), new Example(Tensors.create(55.0, 30.0, 50.0, 80.0, 300.0), Tensors.create(800.0)), new Example(Tensors.create(60.0, 20.0, 50.0, 80.0, 100.0), Tensors.create(900.0)), new Example(Tensors.create(75.0, 30.0, 50.0, 80.0, 300.0), Tensors.create(300.0)))

  override def read_chunk(chunk_index: Long): ExampleVector = exampleVec

  override def chunk_count: Long = 1

  override def reset(): Unit = {}

  def apply(exampleVector: ExampleVector): ExampleVector = {

    this.exampleVec = exampleVector
    println(s"ExampleVectorReader reader example data size ${exampleVector.size()}")
    this.exampleVec
  }

  def apply(tensorSeq: Seq[(Tensor, Tensor)]): ExampleVector = {

    this.exampleVec = new ExampleVector(tensorSeq.map(x => new Example(x._1, x._2)).toArray: _*)
    this.exampleVec
  }

}

//    val dataTensor = exampleVector.begin().get().data()
//    val targetTensor = exampleVector.begin().get().target()
//    val dataTensorVector = dataTensor.chunk(batch)
//    val targetTensorVector = targetTensor.chunk(batch)
//    val data = dataTensorVector.begin()
//    val target = targetTensorVector.begin()
//    println("try to split data")
//    val exampleVec = new ExampleVector()
//    var putIndex = 0
//    while ( ! data.equals(dataTensorVector.end()) && ! target.equals(targetTensorVector.end())){
////      println("begin to put")
//      val example = new Example(data.get(),target.get())
////      println("try to put")
//      exampleVec.put(example)
//      putIndex += 1
////      println(s"put finish  ${putIndex}")
//      data.increment()
//      target.increment()
//    }
