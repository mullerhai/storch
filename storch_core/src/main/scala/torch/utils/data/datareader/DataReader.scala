package torch.utils.data.datareader

import org.bytedeco.pytorch.{
  Tensor,
  ExampleOptional,
  Example,
  ChunkDataReader as CDR,
  ExampleStack,
  ExampleVector,
  TensorExampleVector
}
trait DataReader {

//  def read_chunk(chunk_index: Long) =
  def read_chunk(chunk_index: Long): ExampleVector | TensorExampleVector

  def chunk_count: Long = 1

  def reset(): Unit = {}

}
