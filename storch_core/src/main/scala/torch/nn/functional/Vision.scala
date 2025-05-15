package torch
package nn
package functional

import Derive.derive
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  kZeros,
  kBorder,
  kReflection,
  GridSampleMode,
  GridSamplePaddingMode,
  GridSampleFuncOptions,
  LongArrayRef,
  kNearest,
  kLinear,
  kBilinear,
  kBicubic,
  kTrilinear,
  kArea,
  kNearestExact,
  BoolOptional,
  LongVector,
  DoubleVector,
  LongVectorOptional,
  DoubleVectorOptional,
  InterpolateMode,
  PixelShuffleOptions,
  InterpolateFuncOptions,
  InstanceNormFuncOptions,
  LayerNormFuncOptions,
  LocalResponseNormOptions,
  NormalizeFuncOptions,
  ScalarTypeOptional
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}

private[torch] trait Vision {

  def pixel_shuffle[D <: DType](x: Tensor[D], upscale_factor: Int): Tensor[D] = {

    val options = new PixelShuffleOptions(upscale_factor.toLong)
    val result = torchNative.pixel_shuffle(x.native, options)
    fromNative(result)
  }

  def pixel_unshuffle[D <: DType](x: Tensor[D], downscale_factor: Int): Tensor[D] = {
    val result = torchNative.pixel_unshuffle(x.native, downscale_factor)
    fromNative(result)
  }

  def pad[D <: DType](x: Tensor[D], output_size: List[Long]): Tensor[D] = {

    val outputRef = LongArrayRef(output_size.toArray, output_size.length)
    val result = torchNative.pad(x.native, outputRef)
    fromNative(result)
  }

  def interpolate[D <: DType](
      x: Tensor[D],
      scale_factor: List[Double],
      mode: String,
      antialias: Boolean,
      align_corners: Boolean,
      recompute_scale_factor: Boolean,
      output_size: List[Long]
  ): Tensor[D] = {
    val options = new InterpolateFuncOptions
    val nativeMode = mode match {
      case "nearest" | "Nearest"                                                => new kNearest
      case "bilinear" | "Bilinear"                                              => new kBilinear
      case "bicubic" | "Bicubic"                                                => new kBicubic
      case "trilinear" | "Trilinear"                                            => new kTrilinear
      case "area" | "Area" | "kArea" | "KArea"                                  => new kArea
      case "nearestexact" | "nearestExact" | "kNearestExact" | "KNearest|Exact" => new kNearest
      case "linear" | "Linear" | "kLinear" | "KLinear"                          => new kLinear
    }
    options.mode().put(InterpolateMode(nativeMode))

    val lVec = new LongVector(output_size: _*)
    options.size().put(LongVectorOptional(lVec))
    val dVec = new DoubleVector(scale_factor: _*)

    options.scale_factor().put(DoubleVectorOptional(dVec))
    options.align_corners().put(BoolOptional(align_corners))
    options.recompute_scale_factor().put(BoolOptional(recompute_scale_factor))

    options.antialias().put(antialias)
    val result = torchNative.interpolate(x.native, options)
    fromNative(result)
  }

//  def upsample[D <: DType](x:Tensor[D],output_size:List[Int]):Tensor[D] = {
//    val result = torchNative.upsample(x.native, output_size)
//    fromNative(result)
//  }

//  def upsample_nearest[D <: DType](x:Tensor[D],output_size:List[Int]):Tensor[D] = {
//    val result = torchNative.upsample_nearest(x.native, output_size)
//    fromNative(result)
//
//  }
//
//  def upsample_bilinear[D <: DType](x:Tensor[D],output_size:List[Int]):Tensor[D] = {
//    val result = torchNative.upsample_bilinear(x.native, output_size)
//    fromNative(result)
//  }
  def grid_sample[D <: DType](
      x: Tensor[D],
      grid: Tensor[D],
      mode: String = "bilinear",
      padding_mode: String = "zeros",
      align_corners: Boolean = false
  ): Tensor[D] = {
    val options = GridSampleFuncOptions()
    val nativeMode = mode match {
      case "nearest" | "Nearest"   => new kNearest
      case "bilinear" | "Bilinear" => new kBilinear
    }
    options.mode().put(GridSampleMode(nativeMode))
    val nativePaddingMode = padding_mode match {
      case "zeros" | "Zeros"           => new kZeros
      case "border" | "Border"         => new kBorder
      case "reflection" | "Reflection" => new kReflection

    }
    options.padding_mode().put(nativePaddingMode)
    options.align_corners().put(BoolOptional(align_corners))
    val result = torchNative.grid_sample(x.native, grid.native, options)
    fromNative(result)
  }

  def affine_grid[D <: DType](
      theta: Tensor[D],
      size: List[Long],
      align_corners: Option[Boolean]
  ): Tensor[D] = {

//     val longVec= LongVector(size:_*)
    val longVecRef = LongArrayRef(size.toArray, size.length)
    val result = align_corners match {
      case Some(s) => torchNative.affine_grid(theta.native, longVecRef, s)
      case None    => torchNative.affine_grid(theta.native, longVecRef)

    }

    // val result = torchNative.affine_grid(theta.native, longVecRef)
    fromNative(result)
  }

//  def upsample_bicubic[D <: DType](x:Tensor[D],output_size:List[Int]):Tensor[D] = {
//    val result = torchNative.upsample_bicubic(x.native, output_size)
//    fromNative(result)
//  }
}
