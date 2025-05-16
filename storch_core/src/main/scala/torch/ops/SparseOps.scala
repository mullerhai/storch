package torch
package ops

import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{BoolOptional, DeviceOptional, LayoutOptional, ScalarTypeOptional}
import torch.Device.CPU
import torch.Derive
import torch.Layout.{Strided,Sparse,SparseCsc,SparseBsc,SparseBsr,SparseCsr}
trait SparseOps {



  //https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
  def sparse_coo_tensor[D1 <: DType, D2 <: SparseIntNN](indices: Tensor[D2],
                                                  values: Tensor[D1]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.sparse_coo_tensor(indices.native, values.native))

  def sparse_coo_tensor[D1 <: DType, D2 <: SparseIntNN](indices: Tensor[D2],
                                                  values: Tensor[D1],
                                                  sizeSeq: Seq[Int]): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.sparse_coo_tensor(indices.native, values.native, sizeSeq.map(_.toLong) *))

  def sparse_coo_tensor[D1 <: DType, D2 <: SparseIntNN](
                                                  indices: Tensor[D2],
                                                  values: Tensor[D1],
                                                  dtype: DType|Option[DType] = DType.float32,
                                                  layout: Layout = Sparse,
                                                  device: Device = CPU,
                                                  requires_grad: Boolean = false
                                                 ): Tensor[Promoted[D1, D2]] = {
    val derivedDType = dtype match
      case None => values.dtype
      case d: Option[DType]  => d.get
      case d: DType => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    fromNative(torchNative.sparse_coo_tensor(indices.native, values.native, scalarNative, layoutNative, deviceNative, boolOption,boolOption))
  }


  //https://pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html
  def sparse_csc_tensor[D1 <: DType, D2 <: SparseIntNN](
                                                  ccol_indices: Tensor[D2],
                                                  row_indices: Tensor[D2],
                                                  values: Tensor[D1],
                                                  size: Seq[Int],
                                                  dtype: DType|Option[DType] = DType.float32,
                                                  layout: Layout = SparseCsc,
                                                  device: Device = CPU,
                                                  requires_grad: Boolean = false
                                                 ): Tensor[Promoted[D1, D2]] = {
    val derivedDType = dtype match
      case None => values.dtype
      case d: Option[DType]  => d.get
      case d: DType => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    fromNative(torchNative.sparse_csc_tensor(ccol_indices.native, row_indices.native, values.native,size.map(_.toLong).toArray, scalarNative, layoutNative, deviceNative, boolOption))
  }

  def is_sparse[D1 <: DType](tensor: Tensor[D1]):Boolean = {
    val sparseLayout = Array(Sparse,SparseCsc,SparseBsc,SparseBsr,SparseCsr)
    if sparseLayout.contains(tensor.layout) then true else false
  }

  def to_dense[D1 <: DType](tensor: Tensor[D1]):Tensor[D1]={
     fromNative(tensor.native.to_dense)
  }

  //https://pytorch.org/docs/stable/generated/torch.sparse_csr_tensor.html
  def sparse_csr_tensor[D1 <: DType, D2 <: SparseIntNN](
                                                  crow_indices: Tensor[D2],
                                                  col_indices: Tensor[D2],
                                                  values: Tensor[D1],
                                                  dtype: DType|Option[DType] = DType.float32,
                                                  layout: Layout = SparseCsr,
                                                  device: Device = CPU,
                                                  requires_grad: Boolean = false
                                                 ): Tensor[Promoted[D1, D2]] = {
    val derivedDType = dtype match
      case None => values.dtype
      case d: Option[DType]  => d.get
      case d: DType => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)

    fromNative(torchNative.sparse_csr_tensor(crow_indices.native, col_indices.native, values.native, scalarNative, layoutNative, deviceNative, boolOption))
  }

  //https://pytorch.org/docs/stable/generated/torch.sparse_bsc_tensor.html
  def sparse_bsc_tensor[D1 <: DType, D2 <: SparseIntNN](
                                                  ccol_indices: Tensor[D2],
                                                  row_indices: Tensor[D2],
                                                  values: Tensor[D1],
                                                  size: Seq[Int],
                                                  dtype: DType|Option[DType] = DType.float32,
                                                  layout: Layout = SparseBsc,
                                                  device: Device = CPU,
                                                  requires_grad: Boolean = false
                                                 ): Tensor[Promoted[D1, D2]] = {

    val derivedDType = dtype match
//      case _: Derive => values.dtype
      case None => values.dtype
      case d: Option[DType]  => d.get
      case d: DType => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    fromNative(torchNative.sparse_bsc_tensor(ccol_indices.native, row_indices.native, values.native, size.map(_.toLong).toArray, scalarNative, layoutNative, deviceNative, boolOption))

  }


  //    https: //pytorch.org/docs/stable/generated/torch.sparse_bsr_tensor.html
  def sparse_bsr_tensor[D1 <: DType, D2 <: SparseIntNN](
                                                  crow_indices: Tensor[D2],
                                                  col_indices: Tensor[D2],
                                                  values: Tensor[D1],
                                                  size: Seq[Int],
                                                  dtype: DType|Option[DType] = DType.float32,
                                                  layout: Layout = SparseBsr,
                                                  device: Device = CPU,
                                                  requires_grad: Boolean = false
                                                 ): Tensor[Promoted[D1, D2]] = {
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    val derivedDType = dtype match
//      case _: Derive => values.dtype
      case None => values.dtype
      case d: Option[DType]  => d.get
      case d: DType => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    fromNative(torchNative.sparse_bsr_tensor(crow_indices.native, col_indices.native, values.native,size.map(_.toLong).toArray, scalarNative, layoutNative, deviceNative, boolOption))
  }

  //https://pytorch.org/docs/stable/generated/torch.sparse_compressed_tensor.html
  def sparse_compressed_tensor[D1 <: DType, D2 <: SparseIntNN](
                                                         compressed_indices: Tensor[D2],
                                                         plain_indices: Tensor[D2],
                                                         values: Tensor[D1],
                                                         size: Seq[Int],
                                                         dtype: DType|Option[DType] = DType.float32,
                                                         layout: Layout = Sparse,
                                                         device: Device = CPU,
                                                         requires_grad: Boolean = false
                                                        ): Tensor[Promoted[D1, D2]] = {
    val derivedDType = dtype match
//      case _: Derive => values.dtype
      case None => values.dtype
      case d: Option[DType]  => d.get
      case d: DType => d
    val scalarNative =
      if dtype == values.dtype then ScalarTypeOptional()
      else ScalarTypeOptional(derivedDType.toScalarType)
    val layoutNative: LayoutOptional = LayoutOptional(layout.toNative)
    val deviceNative: DeviceOptional = DeviceOptional(device.toNative)
    val boolOption = BoolOptional(requires_grad)
    fromNative(torchNative.sparse_compressed_tensor(compressed_indices.native, plain_indices.native, values.native, size.map(_.toLong).toArray, scalarNative, layoutNative, deviceNative, boolOption))
  }

  def sparse_mask_out[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2],
                                                t3: Tensor[D1 | D2]
                                               ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.sparse_mask_out(t1.native, t2.native, t3.native))


}

//  def sparse_bsc_tensor[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))
//
//  def sparse_bsr_tensor[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))
//
//
//
//  def sparse_bsc_tensor[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))
//
//  def sparse_bsr_tensor[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))





//  def lstm_cell[D1 <: DType, D2 <: DType](t1: Tensor[D1],vec:Seq[Tensor[D2]] t2: Tensor[D2], t3: Tensor[D1|D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.lstm_cell(t1.native, t2.native, t3.native))
//
//  def gru_cell[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2], t3: Tensor[D1|D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.gru_cell(t1.native, t2.native, t3.native))
//
//  def quantized_gru_cell[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2], t3: Tensor[D1|D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.quantized_gru_cell(t1.native, t2.native, t3.native))










//  def col2im[D1 <: DType](t1: Tensor[D1], sliceSeq: Long): Tensor[D1] =
//    fromNative(torchNative.col2im(t1.native, sliceSeq))








//  def logical_or[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))
//
//  def logical_or[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))
//
//  def logical_or[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))
//
//  def logical_or[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))
//
//  def logical_or[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(torchNative.logical_or(t1.native, t2.native))
//
//  def log10[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.log10(t1.native))
//
//  def log10[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.log10(t1.native))
//
//  def log10[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
//    fromNative(torchNative.log10(t1.native))

  //  public static native T_TensorTensor_T multi_head_attention_forward(
  //  @Const @ByRef Tensor var0
  //  , @Const @ByRef Tensor var1
  //  , @Const @ByRef Tensor var2
  //  , @Const @ByRef MultiheadAttentionForwardFuncOptions var3
  //  );





//  @native def sparse_csc_tensor(@Const @ByRef var0: Tensor,
//                                @Const @ByRef var1: Tensor,
//                                @Const @ByRef var2: Tensor,
//                                @ByVal @Cast(Array(Array()) var3: Array[Long],
//                                @ByVal var4: ScalarTypeOptional,
//                                @ByVal var5: LayoutOptional,
//                                @ByVal var6: DeviceOptional,
//                                @ByVal var7: BoolOptional): Tensor
//
//  def sparse_csr_tensor(@Const @ByRef Tensor var0,
//                        @Const @ByRef Tensor var1,
//                        @Const @ByRef Tensor var2,
//                        @ByVal ScalarTypeOptional var3,
//                        @ByVal LayoutOptional var4,
//                        @ByVal DeviceOptional var5,
//                        @ByVal BoolOptional var6);
//  def sparse_bsc_tensor(@Const @ByRef Tensor var0,
//                        @Const @ByRef Tensor var1,
//                        @Const @ByRef Tensor var2,
//                        @ByVal @StdVector("int64_t") long []var3
//  ,                     @ByVal ScalarTypeOptional var4
//  ,                     @ByVal LayoutOptional var5
//  ,                      @ByVal DeviceOptional var6
//  ,                     @ByVal BoolOptional var7);
//
//  def sparse_bsr_tensor(@Const @ByRef Tensor var0,
//                        @Const @ByRef Tensor var1,
//                        @Const @ByRef Tensor var2,
//                        @ByVal@StdVector("int64_t") long[] var3,
//                        @ByVal ScalarTypeOptional var4,
//                        @ByVal LayoutOptional var5,
//                        @ByVal DeviceOptional var6,
//                        @ByVal BoolOptional var7);
//
//  def  sparse_compressed_tensor(
//  @Const @ByRef Tensor var0
//  , @Const @ByRef Tensor var1
//  , @Const @ByRef Tensor var2
//  }) @StdVector("int64_t") long[] var3
//  , @ByVal ScalarTypeOptional var4
//  , @ByVal LayoutOptional var5
//  , @ByVal DeviceOptional var6
//  , @ByVal BoolOptional var7
//  );
//
//
//
//def sparse_csc_tensor(@Const @ByRef var0: Tensor,
//                      @Const @ByRef var1: Tensor,
//                      @Const @ByRef var2: Tensor,  @StdVector("int64_t") var3: Array[Long],
//                      @ByVal var4: ScalarTypeOptional,
//                      @ByVal var5: LayoutOptional,
//                      @ByVal var6: DeviceOptional,
//                      @ByVal var7: BoolOptional)
//
//
//def sparse_coo_tensor(
//@Const @ByRef Tensor var0
//, @Const @ByRef Tensor var1
//, @ByVal ScalarTypeOptional var2
//, @ByVal LayoutOptional var3
//, @ByVal DeviceOptional var4
//, @ByVal BoolOptional var5
//, @ByVal BoolOptional var6
//);
//
