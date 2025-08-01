package torch.utils.data.dataset

//import java.nio.file.Paths
import torch.{DType, Default, Int64, Tensor}
import scala.collection.Iterator
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

//  TargetType <: DType :Int64
//trait DType
trait Dataset[ParamType <: DType: Default] {
  //  def init(data: AnyRef*): Unit

  def length: Long

  def getItem(idx: Int): (Tensor[ParamType], Tensor[Int64])

  //  def apply(data:AnyRef*):Unit =
  //    init(data)
}
