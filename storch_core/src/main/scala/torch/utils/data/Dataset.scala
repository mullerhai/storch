package torch
package utils
package data

import org.bytedeco.pytorch.{Example, ExampleVector, ExampleIterator}


trait Dataset extends NormalDataset {

  def length: Long

  def getItem(idx: Int): (Tensor[? <: DType], Tensor[? <: DType])

  def get_item(idx: Int): (Tensor[? <: DType], Tensor[? <: DType]) = getItem(idx)
  
  def get_batch(request: Seq[Long]): ExampleVector //= super.get_batch(request*)

  def iterator: Iterator[(Tensor[? <: DType], Tensor[? <: DType])]

}

















  //with Iterable[(Tensor[? <: DType], Tensor[? <: DType])]

  //  override def iterator: Iterator[(Tensor[? <: DType], Tensor[? <: DType])] =
//    new Iterator[(Tensor[? <: DType], Tensor[? <: DType])] {
//
//      private var current: ExampleIterator = nativeDataLoader.begin
//
//      private val endIterator: ExampleIterator = nativeDataLoader.end
//
//      override def hasNext: Boolean = !current.equals(endIterator)
//
//      override def next(): (Tensor[ParamType], Tensor[ParamType]) = {
//        val batch = current.access
//        current = current.increment
//        exampleToTuple(batch)
//      }
//    }


//  override def size(): Int = super.size


//  def init(data: AnyRef*): Unit
//  def apply(data:AnyRef*):Unit =
//    init(data)
