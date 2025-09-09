package torch
package utils
package data

//  TargetType <: DType :Int64
//trait DType
trait Dataset[ParamType <: DType: Default] extends NormalDataset {
  //  def init(data: AnyRef*): Unit

  def length: Long

  def getItem(idx: Int): (Tensor[ParamType], Tensor[Int64])

  //  def apply(data:AnyRef*):Unit =
  //    init(data)
}
