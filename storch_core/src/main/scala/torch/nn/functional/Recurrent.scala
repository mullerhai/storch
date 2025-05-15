package torch
package nn
package functional

import Derive.derive
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Tensor as TensorNative,
  LongOptional,
  PackedSequence,
  MultiheadAttentionForwardFuncOptions,
  TensorVector,
  TensorOptional,
  TensorArrayRef
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.{fromNative, toArrayRef}
import org.bytedeco.pytorch.ScalarTypeOptional
import org.bytedeco.javacpp.{Pointer, CLongPointer}
import org.bytedeco.javacpp.annotation.{Const, ByRef, ByVal, Namespace}
import torch.internal.NativeConverters.*

//case class TensorTriple[T <: FloatNN | ComplexNN](
private[torch] trait Recurrent {

  //  def conv3ds[D <: FloatNN | ComplexNN](
  //                                        input: Tensor[D],
  //                                        weight: Tensor[D],
  //                                        bias: Tensor[D] | Option[Tensor[D]] = None,
  //                                        stride: Int = 1,
  //                                        padding: Int = 0,
  //                                        dilation: Int = 1,
  //                                        groups: Int = 1
  //                                      ): Tensor[D] =
  //    fromNative(
  //      torchNative.conv3d(
  //        input.native,
  //        weight.native,
  //        toOptional(bias),
  //        Array(stride.toLong),
  //        Array(padding.toLong),
  //        Array(dilation.toLong),
  //        groups
  //      )
  //    )

  //  public static native T_TensorTensorTensor_T lstm(
  // @Const @ByRef Tensor var0,
  //  @ByVal TensorArrayRef var1, @ByVal TensorArrayRef var2,

  //  @Cast({"bool"}) boolean var3, @Cast({"int64_t"}) long var4,
  //  double var6, @Cast({"bool"}) boolean var8,
  //  @Cast({"bool"}) boolean var9, @Cast({"bool"}) boolean var10);
  //
  //  @Namespace("at")
  //  @ByVal
  //  public static native T_TensorTensorTensor_T lstm(@Const @ByRef
  //  Tensor var0, @ByVal TensorVector var1, @ByVal TensorVector var2, @Cast({"bool"})
  //  boolean var3, @Cast({"int64_t"}) long var4, double var6, @Cast({"bool"})
  //  boolean var8, @Cast({"bool"}) boolean var9, @Cast({"bool"}) boolean var10);
  //
  //  @Namespace("at")
  //  @ByVal
  //  public static native T_TensorTensorTensor_T lstm(@Const @ByRef Tensor var0,
  //  @Const @ByRef Tensor var1, @ByVal TensorArrayRef var2, @ByVal TensorArrayRef var3,
  //  @Cast({"bool"}) boolean var4, @Cast({"int64_t"}) long var5, double var7,
  //  @Cast({"bool"}) boolean var9, @Cast({"bool"}) boolean var10);
  //
  //  @Namespace("at")
  //  @ByVal
  //  public static native T_TensorTensorTensor_T lstm(@Const @ByRef
  //  Tensor var0, @Const @ByRef Tensor var1, @ByVal TensorVector var2,
  //  @ByVal TensorVector var3, @Cast({"bool"}) boolean var4, @Cast({"int64_t"})
  //  long var5, double var7, @Cast({"bool"}) boolean var9, @Cast({"bool"}) boolean var10);

  // TensorArrayRef(@Const Tensor data, @Cast("size_t") long length)

  //  lstm_cell[T, TT <: DType](@Const @ByRef
  //  input: Tensor[T, TT], @ByVal hx: TensorList[T, TT], @Const @ByRef w_ih: Tensor[T, TT],
  //  @Const @ByRef w_hh: Tensor[T, TT]): TensorTuple[T,T,TT]

  //	@native @Namespace("at::native") @ByVal def lstm_cell[T, TT <: DType](@Const @ByRef
  //	input: Tensor[T, TT], @ByVal hx: TensorList[T, TT], @Const @ByRef w_ih: Tensor[T, TT],
  //	@Const @ByRef w_hh: Tensor[T, TT], @Const @ByRef
  //	b_ih: Tensor[T, TT], @Const @ByRef b_hh: Tensor[T, TT]): TensorTuple[T,T,TT]

  def gru_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT],
      b_ih: Tensor[TT],
      b_hh: Tensor[TT]
  ): Tensor[TT] = {

    val inputNative = input.native
    val hxNative = hx.native // TensorVector(hx.map(_.native) *)
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    val b_ihNative = TensorOptional(b_ih.native)
    val b_hhNative = TensorOptional(b_hh.native)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native =
      torchNative.gru_cell(inputNative, hxNative, w_ihNative, w_hhNative, b_ihNative, b_hhNative)
    //    TensorTuple2(output = fromNative[TT](native.get0()), hx = fromNative(native.get1))

    //    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    fromNative(native)
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def multi_head_attention_forward[T, D, TT <: FloatNN | ComplexNN](
      query: Tensor[TT],
      key: Tensor[TT],
      value: Tensor[TT],
      embed_dim_to_check: Long,
      num_heads: Long,
      in_proj_weight: Tensor[TT],
      in_proj_bias: Tensor[TT],
      bias_k: Tensor[TT],
      bias_v: Tensor[TT],
      add_zero_attn: Boolean,
      dropout_p: Double,
      out_proj_weight: Tensor[TT],
      out_proj_bias: Tensor[TT],
      training: Boolean,
      key_padding_mask: Tensor[TT],
      need_weights: Boolean,
      attn_mask: Tensor[TT],
      use_separate_proj_weight: Boolean,
      q_proj_weight: Tensor[TT],
      k_proj_weight: Tensor[TT],
      v_proj_weight: Tensor[TT],
      static_k: Tensor[TT],
      static_v: Tensor[TT],
      average_attn_weights: Boolean
  ): TensorTuple2[TT] = {

    val queryNative = query.native
    val keyNative = key.native
    val valueNative = value.native
    val options = MultiheadAttentionForwardFuncOptions(
      embed_dim_to_check,
      num_heads,
      in_proj_weight.native,
      in_proj_bias.native,
      bias_k.native,
      bias_v.native,
      add_zero_attn,
      dropout_p,
      out_proj_weight.native,
      out_proj_bias.native
    )

    options.training().put(training)
    options.key_padding_mask().put(key_padding_mask.native)
    options.need_weights().put(need_weights)
    options.attn_mask().put(attn_mask.native)
    options.use_separate_proj_weight().put(use_separate_proj_weight)
    options.average_attn_weights().put(average_attn_weights)
    options.static_k().put(static_k.native)
    options.static_v().put(static_v.native)
    options.q_proj_weight().put(q_proj_weight.native)
    options.k_proj_weight().put(k_proj_weight.native)
    options.v_proj_weight().put(v_proj_weight.native)

    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native =
      torchNative.multi_head_attention_forward(queryNative, keyNative, valueNative, options)
    //    TensorTuple(values = fromNative[TT](native.get0()), indices = fromNative(native.get1))

    val tensorTuple2 = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple2
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  //  inline Tensor torch::nn::utils::rnn:
  //  :invert_permutation(const Tensor &permutation)
  def invert_permutation[D <: DType](permutation: Tensor[D]): Tensor[D] = {

    fromNative(torchNative.invert_permutation(permutation.native))
  }
  // inline PackedSequence torch :: nn :: utils :: rnn :: pack_padded_sequence(
  //   Tensor input, Tensor lengths, bool batch_first = false, bool enforce_sorted = true)

  //  def pack_padded_sequence[D<:DType](input:Tensor[D],lengths:Tensor[D],batch_first:Boolean,enforce_sorted:Boolean):Tensor[D]={
  //
  //  }
  //
  //  def pack_sequence[T, TT <: FloatNN | ComplexNN](sequences: Seq[Tensor[TT]],enforce_sorted:Boolean):TensorTuple[D]={
  //
  //  }
  //
  //  def pad_packed_sequence[T,D, TT <: FloatNN | ComplexNN](sequence:Seq[Tensor[D]],batch_first:Boolean,padding_value:Double,total_length:Long):TensorTuple[D]={
  //
  //  }

  // inline PackedSequence torch::nn::utils::rnn::pack_sequence(
  //   ArrayRef<Tensor> sequences, bool enforce_sorted = true)

  // inline std::tuple<Tensor, Tensor> torch::nn::utils::rnn::pad_packed_sequence(
  //   const PackedSequence &sequence, bool batch_first = false,
  //   double padding_value = 0.0, std::optional<int64_t> total_length = std::nullopt)

  //  inline Tensor torch::nn::utils::rnn::pad_sequence
  //  (ArrayRef<Tensor> sequences,
  //    bool batch_first = false, double padding_value = 0,
  //    std::string_view padding_side = "right")
  def pad_sequence[T, TT <: FloatNN | ComplexNN](
      sequences: Seq[Tensor[TT]],
      batch_first: Boolean,
      padding_value: Double = 0,
      padding_side: String = "right"
  ): Tensor[TT] = {
    val sequencesNative = toArrayRef(sequences)
    val native = torchNative.pad_sequence(sequencesNative, batch_first, padding_value, padding_side)
    fromNative(native)
  }

  //
  def gru[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Tensor[TT],
      batchSizes: Tensor[TT],
      params: Option[Seq[Tensor[TT]]],
      hasBiases: Boolean,
      numLayers: Long, // CLongPointer,
      dropout: Double,
      train: Boolean,
      bidirectional: Boolean,
      batchFirst: Boolean
  ): TensorTuple2[TT] = {

    val inputNative = input.native
    //    val hxNative = toArrayRef(hx)
    val paramsNtive = toArrayRef(params.get)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native = torchNative.gru(
      inputNative,
      hx.native,
      paramsNtive,
      hasBiases,
      numLayers,
      dropout,
      train,
      bidirectional,
      batchFirst
    )
    //    TensorTuple(values = fromNative[TT](native.get0()), indices = fromNative(native.get1))

    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  //
  def lstm_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Seq[Tensor[TT]],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT],
      b_ih: Tensor[TT],
      b_hh: Tensor[TT]
  ): TensorTuple2[TT] = {

    val inputNative = input.native
    val hxNative = TensorVector(hx.map(_.native)*)
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    val b_ihNative = TensorOptional(b_ih.native)
    val b_hhNative = TensorOptional(b_hh.native)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native =
      torchNative.lstm_cell(inputNative, hxNative, w_ihNative, w_hhNative, b_ihNative, b_hhNative)
    TensorTuple2(output = fromNative[TT](native.get0()), hx = fromNative(native.get1))

    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def lstm_cell[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Seq[Tensor[TT]],
      w_ih: Tensor[TT],
      w_hh: Tensor[TT]
  ): TensorTuple2[TT] = {

    val inputNative = input.native
    val hxNative = toArrayRef(hx)
    val w_ihNative = w_ih.native
    val w_hhNative = w_hh.native
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native = torchNative.lstm_cell(inputNative, hxNative, w_ihNative, w_hhNative)
    TensorTuple2(output = fromNative[TT](native.get0()), hx = fromNative(native.get1))

    val tensorTuple = TensorTuple2(fromNative[TT](native.get0()), fromNative[TT](native.get1()))
    tensorTuple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def lstm[T, TT <: FloatNN | ComplexNN](
      input: Tensor[TT],
      hx: Seq[Tensor[TT]],
      params: Option[Seq[Tensor[TT]]],
      hasBiases: Boolean,
      numLayers: Long, // CLongPointer,
      dropout: Double,
      train: Boolean,
      bidirectional: Boolean,
      batch_first: Boolean
  ): TensorTriple[TT] = {

    val inputNative = input.native
    val hxNative = toArrayRef(hx)
    val paramsNtive = toArrayRef(params.get)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native = torchNative.lstm(
      inputNative,
      hxNative,
      paramsNtive,
      hasBiases,
      numLayers,
      dropout,
      train,
      bidirectional,
      batch_first
    )
    //    TensorTuple(values = fromNative[TT](native.get0()), indices = fromNative(native.get1))

    val tensorTriple = TensorTriple(
      fromNative[TT](native.get0()),
      fromNative[TT](native.get1()),
      fromNative[TT](native.get2())
    )
    tensorTriple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }

  def lstm[T, TT <: FloatNN | ComplexNN](
      data: Tensor[TT],
      batchSize: Int,
      hx: Seq[Tensor[TT]],
      params: Option[Seq[Tensor[TT]]],
      hasBiases: Boolean,
      numLayers: Long, // CLongPointer,
      dropout: Double,
      train: Boolean,
      bidirectional: Boolean,
      batch_first: Boolean
  ): TensorTriple[TT] = {

    val dataNative = data.native
    val hxNative = toArrayRef(hx)
    val batchSizeNative = torchNative.tensor(batchSize)
    val paramsNtive = toArrayRef(params.get)
    //    val numLayersNative = CLongPointer( numLayers.toArray:_*)
    val native = torchNative.lstm(
      dataNative,
      batchSizeNative,
      hxNative,
      paramsNtive,
      hasBiases,
      numLayers,
      dropout,
      train,
      bidirectional
    )
    //  TensorTuple(values = fromNative[TT](native.get0()), indices = fromNative(native.get1))

    val tensorTriple = TensorTriple(
      fromNative[TT](native.get0()),
      fromNative[TT](native.get1()),
      fromNative[TT](native.get2())
    )
    tensorTriple
    //    fromNative(torchNative.lstm(inputNative, hxNative, paramsNtive, has_biases, num_layers, dropout, train, bidirectional, batch_first))

  }
  //

}

case class TensorTuple2[T <: FloatNN | ComplexNN](output: Tensor[T], hx: Tensor[T])

case class TensorTriple[T <: FloatNN | ComplexNN](
    output: Tensor[T],
    indices: Tensor[T],
    values: Tensor[T]
)
