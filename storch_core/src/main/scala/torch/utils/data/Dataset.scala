package torch
package utils
package data


trait Dataset extends NormalDataset {
  
  def length: Long

  def getItem(idx: Int): (Tensor[? <: DType], Tensor[Int64])
  
}













//  def init(data: AnyRef*): Unit
//  def apply(data:AnyRef*):Unit =
//    init(data)