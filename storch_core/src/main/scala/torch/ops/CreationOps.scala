/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package ops

import internal.NativeConverters
import NativeConverters.*
import Layout.Strided
import Device.CPU
import MemoryFormat.Contiguous
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  BoolOptional,
  ByteVector,
  GenericDict,
  IValue,
  LongOptional,
  MemoryFormatOptional,
  OutputArchive,
  Scalar,
  ScalarTypeOptional
}
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.global.torch.ScalarType
import torch.nn.modules.Module as ModelModule

import java.nio.file.{Files, Path, Paths}
import scala.collection.immutable.{SeqMap, VectorMap}
import torch.internal.NativeConverters.fromNative

import scala.reflect.ClassTag

/** Creation Ops
  *
  * https://pytorch.org/docs/stable/torch.html#creation-ops
  */
private[torch] trait CreationOps {

// TODO sparse_coo_tensor
// TODO as_tensor
// TODO as_strided
// TODO frombuffer

  def as_tensor[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]],
      requires_grad: Boolean = false
  ): Tensor[ScalaToDType[U]] = Tensor.apply(data, requires_grad)

  def tensor[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]],
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[ScalaToDType[U]] = Tensor.apply(data, layout, device, requires_grad)
  def tensor[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]],
      requires_grad: Boolean
  ): Tensor[ScalaToDType[U]] = Tensor.apply(data, requires_grad)

  def tensor[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]],
      requires_grad: Boolean,
      device: Device
  ): Tensor[ScalaToDType[U]] = Tensor.apply(data, requires_grad, device)

  def as_strided[D <: DType](
      input: Tensor[D],
      size: Seq[Int],
      stride: Seq[Int],
      storageOffset: Option[Int] = None
  ): Tensor[D] = {
    if storageOffset.isDefined then
      fromNative[D](
        torchNative.as_strided(
          input.native,
          size.map(_.toLong).toArray,
          stride.map(_.toLong).toArray,
          new LongOptional(storageOffset.get.toLong)
        )
      )
    else
      fromNative[D](
        torchNative.as_strided(
          input.native,
          size.map(_.toLong).toArray,
          stride.map(_.toLong).toArray,
          new LongOptional()
        )
      )
  }

  def as_strided[D <: DType](input: Tensor[D], size: Seq[Int], stride: Seq[Int]): Tensor[D] = {
    fromNative[D](
      torchNative.as_strided(input.native, size.map(_.toLong).toArray, stride.map(_.toLong)*)
    )
  }
// def zeros[D <: DType](size: Int*): Tensor[Float32] =
//   zeros[D](size.toSeq)
  def cumsum[D <: DType](input: Tensor[D], dim: Long, dtype: D = float32): Tensor[D] =
    fromNative(
      torchNative.cumsum(input.native, dim, ScalarTypeOptional(dtype.toScalarType))
    )

  /** Returns a tensor filled with the scalar value `0`, with the shape defined by the variable
    * argument `size`.
    *
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    * @tparam T
    * @return
    *
    * @group creation_ops
    */
  def zeros[D <: DType](
      size: Seq[Int] | Int,
      dtype: D,
      requires_grad: Boolean
  ): Tensor[D] = this.zeros(size, dtype, Strided, CPU, requires_grad)

  def ones[D <: DType](
      size: Seq[Int] | Int,
      dtype: D,
      requires_grad: Boolean
  ): Tensor[D] = this.ones(size, dtype, Strided, CPU, requires_grad)

//  def ones[D <: DType](
//                         size: Seq[Int] | Int,
//                         requires_grad: Boolean
//                       ): Tensor[D] = this.ones(size, torch.float32, Strided, CPU, requires_grad)
//
  def zeros[D <: DType](
      size: Seq[Int] | Int,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val nativeSize = size match
      case s: Seq[Int] => s.map(_.toLong).toArray
      case s: Int      => Array(s.toLong)
    fromNative(
      torchNative.torch_zeros(
        nativeSize,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  def zeros_like[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_zeros_like)

  /** @group creation_ops
    */
  def zerosLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_zeros_like)

  /** Returns a tensor filled with the scalar value `1`, with the shape defined by the variable
    * argument `size`.
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    *
    * @group creation_ops
    */
  def ones[D <: DType](
      size: Seq[Int] | Int,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val nativeSize = size match
      case s: Seq[Int] => s.map(_.toLong).toArray
      case s: Int      => Array(s.toLong)
    fromNative(
      torchNative.torch_ones(
        nativeSize,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  def ones_like[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_ones_like)

  /** @group creation_ops */
  def onesLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_ones_like)

// format: off
/** Returns a 1-D tensor of size $`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`$ with values
  * from the interval ``[start, end)`` taken with common difference `step` beginning from `start`.
  *
  * Note that non-integer `step` is subject to floating point rounding errors when comparing against `end`;
  * to avoid inconsistency, we advise adding a small epsilon to `end` in such cases.
  *
  * $$
  * \text{out}_{{i+1}} = \text{out}_{i} + \text{step}
  * $$
  *
  * @param start
  *   The starting value for the set of points. Default: ``0``.
  * @param end
  *   The ending value for the set of points
  * @param step
  *   The gap between each pair of adjacent points. Default: ``1``.
  *   
  * @group creation_ops
  */
// format: on

  def arange[D <: DType | Derive, Start <: ScalaType, End <: ScalaType, Step <: ScalaType](
      start: Start = 0,
      end: End,
      step: Step = 1,
      dtype: D = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[DTypeOrDeriveArange[D, Start, End, Step]] =
    val derivedDType = dtype match
      case _: Derive => derivedArangeType(start, end, step)
      case t: DType  => t
    fromNative(
      torchNative.torch_arange(
        toScalar(start),
        toScalar(end),
        toScalar(step),
        NativeConverters.tensorOptions(derivedDType, layout, device, requires_grad)
      )
    )

  /** @group creation_ops */
  def linspace[D <: DType](
      start: Double,
      end: Double,
      steps: Long,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_linspace(
        new Scalar(start),
        new Scalar(end),
        steps,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  /** @group creation_ops */
  def logspace[D <: DType](
      start: Double,
      end: Float,
      steps: Long,
      base: Double = 10.0,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ) = fromNative(
    torchNative.torch_logspace(
      new Scalar(start),
      new Scalar(end),
      steps,
      base,
      NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    )
  )

  /** Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
    *
    * @param n
    *   the number of rows
    * @param m
    *   the number of columns with default being `n`
    * @param dtype
    *   the desired data type of the returned tensor.
    * @param layout
    *   the desired layout of the returned tensor.
    * @param device
    *   the desired device of the returned tensor.
    * @param requires_grad
    *   If autograd should record operations on the returned tensor.
    *
    * @group creation_ops
    */
  def eye[D <: DType](
      n: Int,
      m: Option[Int] = None,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] = fromNative(
    torchNative.torch_eye(n, NativeConverters.tensorOptions(dtype, layout, device, requires_grad))
  )
// def empty(size: Long*): Tensor[D] = fromNative(torchNative.torch_empty(size*))

  /** Returns a tensor filled with uninitialized data.
    *
    * @group creation_ops
    */
  def empty[D <: DType](
      size: Seq[Int],
      dtype: D,
      requires_grad: Boolean
  ): Tensor[D] = this.empty(size, dtype, Strided, CPU, requires_grad, false, Contiguous)

//  def empty[D <: DType](
//                         size: Seq[Int],
//                         requires_grad: Boolean
//                       ): Tensor[D] = this.empty[Float32](size, float32, Strided, CPU, requires_grad, false, Contiguous)

  def empty[D <: DType](
      size: Seq[Int],
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false,
      pinMemory: Boolean = false,
      memoryFormat: MemoryFormat = Contiguous
  ): Tensor[D] =
    fromNative(
      torchNative.torch_empty(
        size.toArray.map(_.toLong),
        NativeConverters
          .tensorOptions(dtype, layout, device, requires_grad)
          .pinned_memory(BoolOptional(pinMemory)),
        new MemoryFormatOptional(memoryFormat.toNative)
      )
    )

  /** Returns an uninitialized tensor with the same size as input.
    *
    * `torch.empty_like(input)` is equivalent to `torch.empty(input.size(), dtype=input.dtype,
    * layout=input.layout, device=input.device`).
    *
    * @group creation_ops
    */
  def emptyLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_empty_like)

  def empty_like[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_empty_like)

  // // TODO emptyStrided

  /** Creates a tensor of size `size` filled with `fillValue`. The tensor's dtype is inferred from
    * `fillValue`.
    *
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    * @param fillValue
    *   the value to fill the output tensor with.
    * @param dtype
    *   the desired data type of the returned tensor.
    * @param layout
    *   the desired layout of the returned Tensor.
    * @param device
    *   the desired device of the returned tensor.
    * @param requires_grad
    *   If autograd should record operations on the returned tensor.
    * @tparam T
    *   the data type of the returned tensor, or `Default` if the type should be derived from
    *   `fillValue`.
    * @tparam U
    *   the data type of `fillValue`.
    * @return
    *   the newly created tensor.
    *
    * @group creation_ops
    */
  def full[D <: DType | Derive, U <: ScalaType](
      size: Seq[Int],
      fillValue: U,
      dtype: D = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] =
    val derivedDType = dtype match
      case _: Derive => scalaToDType(fillValue)
      case t: DType  => t
    fromNative(
      torchNative.torch_full(
        size.toArray.map(_.toLong),
        toScalar(fillValue),
        NativeConverters.tensorOptions(derivedDType, layout, device, requires_grad)
      )
    )

    // input (Tensor) – float tensor to quantize
    // scales (Tensor) – float 1D tensor of scales to use, size should match input.size(axis)
    // zero_points (int) – integer 1D tensor of offset to use, size should match input.size(axis)
    // axis (int) – dimension on which apply per-channel quantization
    // dtype (torch.dtype) – the desired data type of returned tensor. Has to be one of the quantized dtypes: torch.quint8, torch.qint8, torch.qint32
  // torch.quantize_per_channel(input, scales, zero_points, axis, dtype) → Tensor
///  public static native Tensor quantize_per_channel(
  // @Const @ByRef Tensor var0, @Const @ByRef Tensor var1,
  // @Const @ByRef Tensor var2, @Cast({"int64_t"}) long var3, ScalarType var5);
  // dtypes: torch.quint8, torch.qint8, torch.qint32
  def quantize_per_channel[D <: DType, U <: ScalaType](
      input: Tensor[D],
      scales: Tensor[Float32],
      zero_points: Tensor[Int64],
      axis: Int,
      dtype: ScalarType = ScalarType.QUInt8
  ): Tensor[?] = {

    fromNative(
      torchNative.quantize_per_channel(
        input.native,
        scales.native,
        zero_points.native,
        axis.toLong,
        dtype
      )
    )
  }
//  torch.quantize_per_tensor(input, scale, zero_point, dtype) → Tensor
  //    public static native Tensor quantize_per_tensor(@Const @ByRef Tensor var0, double var1, @Cast({"int64_t"}) long var3, ScalarType var5);
//input (Tensor) – float tensor or list of tensors to quantize
//
//scale (float or Tensor) – scale to apply in quantization formula
//
//zero_point (int or Tensor) – offset in integer value that maps to float zero
//
//dtype (torch.dtype) – the desired data type of returned tensor. Has to be one of the quantized dtypes: torch.quint8, torch.qint8, torch.qint32
//    public static native Tensor quantize_per_tensor(@Const @ByRef Tensor var0,
//    @Const @ByRef Tensor var1, @Const @ByRef Tensor var2, ScalarType var3);
  //// dtypes: torch.quint8, torch.qint8, torch.qint32
  def quantize_per_tensor[D <: DType, U <: ScalaType](
      input: Tensor[D],
      scales: Tensor[Float32],
      zero_points: Tensor[Int64],
      dtype: ScalarType = ScalarType.QUInt8
  ): Tensor[?] = {

    fromNative(
      torchNative.quantize_per_tensor(input.native, scales.native, zero_points.native, dtype)
    )
  }

// //dtypes: torch.quint8, torch.qint8, torch.qint32
  def quantize_per_tensor_single[D <: DType, U <: ScalaType](
      input: Tensor[D],
      scale: Double,
      axis: Int,
      dtype: ScalarType = ScalarType.QUInt8
  ): Tensor[?] = {

    fromNative(torchNative.quantize_per_tensor(input.native, scale, axis.toLong, dtype))
  }

  //  public static native Tensor full_like(
  //  @Const @ByRef Tensor var0,
  //  @Const @ByRef Scalar var1, @ByVal(nullValue = "at::TensorOptions{}") T
  //  ensorOptions var2, @ByVal(nullValue = "std::optional<at::MemoryFormat>(::std::nullopt)")
  //  MemoryFormatOptional var3);

  //    public static native Tensor full_like(
  //    @Const @ByRef Tensor var0, @Const @ByRef Scalar var1);

  //    public static native Tensor full_like(
  //    @Const @ByRef Tensor var0, @Const @ByRef Scalar var1,
  //    @ByVal ScalarTypeOptional var2, @ByVal LayoutOptional var3,
  //    @ByVal DeviceOptional var4, @ByVal BoolOptional var5,
  //    @ByVal MemoryFormatOptional var6);
//torch.full_like(input, fill_value, \*, dtype=None, layout=torch.strided,
  // device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor
//Tensor torch_full_like(@Const
  // @ByRef Tensor var0,
  // @Const @ByRef Scalar var1,
  // @ByVal(TensorOptions var2,
  // " MemoryFormatOptional var3);
  def full_like[D <: DType, D1 <: Derive, U <: ScalaType](
      input: Tensor[D],
      fillValue: U,
      dtype: D1 = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] =
    val derivedDType = dtype match
      case _: Derive => scalaToDType(fillValue)
//      case t: DType => t
    fromNative(
      torchNative.torch_full_like(
        input.native,
        toScalar(fillValue),
        NativeConverters.tensorOptions(derivedDType, layout, device, requires_grad),
        memoryFormat.toNativeOptional
      )
    )

// TODO fullLike
// TODO quantize_per_tensor
// TODO quantize_per_channel
// TODO dequantize
// TODO complex
// TODO polar
// TODO heavside

  def pickleLoad(data: ByteVector): SeqMap[String, Tensor[DType]] =
    val dict: GenericDict = torchNative.pickle_load(data).toGenericDict()
    // We need to extract the members in one go or we risk too early deallocation of native objects here
    val buffer = new Array[(IValue, IValue)](dict.size().toInt)
    val nativeIt = dict.begin()
    for (i <- 0 until buffer.size)
      buffer(i) = (nativeIt.access().key(), nativeIt.access().value())
      nativeIt.increment()
    VectorMap.from(buffer.map { (key, value) =>
      // TODO better error handling
      (key.toStringRef().getString(), fromNative[DType](value.toTensor().clone()))
    })

  def pickle_load(path: Path) = pickleLoad(path)

  def pickleLoad(path: Path): Map[String, Tensor[DType]] =
    val data: ByteVector = Files.readAllBytes(path).asInstanceOf[ByteVector]
    pickleLoad(data)

  def save(model: ModelModule, filePath: String): Unit = {
    val archive = new OutputArchive
    model.save(archive)
    archive.save_to(filePath) // "net.pt"
  }

  // torch.save(model.state_dict(), 'resnet.ckpt')
  def save(state_dict: Map[String, Tensor[?]], checkpoint_filePath: String): Unit = {
//    val archive = new OutputArchive
//    archive.save_to(filePath) //"net.pt"
    val byteArray = SerializeUtils.serialize(state_dict)
    Files.write(Paths.get(checkpoint_filePath), byteArray)
  }
  def pickleSave(tensors: SeqMap[String, Tensor[DType]]) =
    tensors.map { (k, v) =>
      (IValue(k), IValue(v.native))
    }

}

object SerializeUtils {

  import java.io.{ByteArrayOutputStream, ObjectOutputStream}

  def serialize(obj: Any): Array[Byte] = {
    val stream = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(stream)
    oos.writeObject(obj)
    oos.close()
    stream.toByteArray
  }
}
