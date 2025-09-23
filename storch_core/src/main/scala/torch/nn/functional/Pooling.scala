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
package nn
package functional

import org.bytedeco.javacpp.{LongPointer, DoublePointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  AvgPool1dOptions,
  LongOptional,
  DoubleExpandingArrayOptional,
  AvgPool2dOptions,
  FractionalMaxPool2dOptions,
  FractionalMaxPool3dOptions,
  AvgPool3dOptions,
  LongExpandingArrayOptional,
  MaxPool1dOptions,
  MaxUnpool1dFuncOptions,
  MaxUnpool2dFuncOptions,
  MaxPool2dOptions,
  MaxUnpool3dFuncOptions,
  MaxPool3dOptions,
  AdaptiveMaxPool1dOptions,
  AdaptiveMaxPool2dOptions,
  AdaptiveMaxPool3dOptions,
  AdaptiveAvgPool1dOptions,
  AdaptiveAvgPool2dOptions,
  AdaptiveAvgPool3dOptions,
  LPPool1dOptions,
  LPPool2dOptions,
  LPPool3dOptions
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.*

/** @define maxPoolPadding
  *   Implicit negative infinity padding to be added on both sides, must be \>= 0 and \<=
  *   kernel_size / 2.
  * @define dilation
  *   The stride between elements within a sliding window, must be \> 0.
  * @define ceilMode
  *   If `true`, will use *ceil* instead of *floor* to compute the output shape. This ensures that
  *   every element in the input tensor is covered by a sliding window.
  */
private[torch] trait Pooling {

  /** Applies a 1D max pooling over an input signal composed of several input planes.
    *
    * @see
    *   [[torch.nn.AdaptiveAvgPool1d]] for details and output shape.
    * @param input
    *   input tensor of shape $(\text{minibatch} , \text{in\_channels} , iW)$
    * @param kernelSize
    *   the size of the window.
    * @param stride
    *   the stride of the window. Default: `kernelSize`
    * @param padding
    *   implicit zero paddings on both sides of the input. Can be a single number or a tuple
    *   `(padW,)`.
    * @param ceilMode
    *   $ceilMode
    * @param countIncludePad
    *   when true, will include the zero-padding in the averaging calculation.
    * @group nn_pooling
    */
  def avg_pool1d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int),
      stride: Int | None.type = None,
      padding: Int = 0,
      ceil_mode: Boolean = false,
      count_include_pad: Boolean = true
  ): Tensor[D] =
    val options = AvgPool1dOptions(toNative(kernel_size))
    stride match
      case s: Int => options.stride().put(toNative(s))
      case None   =>
    options.padding().put(toNative(padding))
    options.ceil_mode().put(ceil_mode)
    options.count_include_pad().put(count_include_pad)
    fromNative(torchNative.avg_pool1d(input.native, options))

  /** Applies a 2D max pooling over an input signal composed of several input planes.
    *
    * @group nn_pooling
    */
  def avg_pool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int) | None.type = None,
      padding: Int | (Int, Int) = 0,
      ceil_mode: Boolean = false,
      count_include_pad: Boolean = true,
      divisor_override: Int | None.type = None
  ): Tensor[D] =
    val options = AvgPool2dOptions(toNative(kernel_size))
    stride match
      case s: (Int | (Int, Int)) => options.stride().put(toNative(s))
      case None                  =>
    options.padding().put(toNative(padding))
    options.ceil_mode().put(ceil_mode)
    options.count_include_pad().put(count_include_pad)
    divisor_override match
      case d: Int => options.divisor_override().put(d)
      case None   =>
    fromNative(torchNative.avg_pool2d(input.native, options))

  /** Applies a 3D max pooling over an input signal composed of several input planes.
    *
    * @group nn_pooling
    */
  def avg_pool3d[D <: Float16 | Float32 | Float64 | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | None.type = None,
      padding: Int | (Int, Int, Int) = 0,
      ceil_mode: Boolean = false,
      count_include_pad: Boolean = true,
      divisor_override: Int | None.type = None
  ): Tensor[D] =
    val options = AvgPool3dOptions(toNative(kernel_size))
    stride match
      case s: (Int | (Int, Int, Int)) => options.stride().put(toNative(s))
      case None                       =>
    options.padding().put(toNative(padding))
    options.ceil_mode().put(ceil_mode)
    options.count_include_pad().put(count_include_pad)
    divisor_override match
      case d: Int => options.divisor_override().put(d)
      case None   =>
    fromNative(torchNative.avg_pool3d(input.native, options))

  private def maxPool1dOptions[D <: FloatNN | Complex32](
      kernelSize: Int | (Int, Int),
      stride: Int | None.type,
      padding: Int,
      dilation: Int,
      ceilMode: Boolean
  ): MaxPool1dOptions =
    val options: MaxPool1dOptions = MaxPool1dOptions(toNative(kernelSize))
    stride match
      case s: Int => options.stride().put(toNative(s))
      case None   =>
    options.padding().put(toNative(padding))
    options.dilation().put(toNative(dilation))
    options.ceil_mode().put(ceilMode)
    options

  /** Applies a 1D max pooling over an input signal composed of several input planes.
    *
    * @param input
    *   input tensor of shape $(\text{minibatch} , \text{in\_channels} , iW)$, minibatch dim
    *   optional.
    * @param kernelSize
    *   the size of the window.
    * @param stride
    *   the stride of the window.
    * @param padding
    *   $maxPoolPadding
    * @param dilation
    *   $dilation
    * @param ceilMode
    *   $ceilMode
    * @group nn_pooling
    */
  def max_pool1d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int),
      stride: Int | None.type = None,
      padding: Int = 0,
      dilation: Int = 1,
      ceil_mode: Boolean = false
  ): Tensor[D] =
    val options: MaxPool1dOptions =
      maxPool1dOptions(kernel_size, stride, padding, dilation, ceil_mode)
    fromNative(torchNative.max_pool1d(input.native, options))

  /** Applies a 1D max pooling over an input signal composed of several input planes.
    *
    * @param input
    *   input tensor of shape $(\text{minibatch} , \text{in\_channels} , iW)$, minibatch dim
    *   optional.
    * @param kernelSize
    *   the size of the window.
    * @param stride
    *   the stride of the window.
    * @param padding
    *   $maxPoolPadding
    * @param dilation
    *   $dilation
    * @param ceilMode
    *   $ceilMode
    * @group nn_pooling
    */
  def max_pool1d_with_indices[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int),
      stride: Int | None.type = None,
      padding: Int = 0,
      dilation: Int = 1,
      ceil_mode: Boolean = false
  ): TensorTuple[D] =
    val options: MaxPool1dOptions =
      maxPool1dOptions(kernel_size, stride, padding, dilation, ceil_mode)
    val native = torchNative.max_pool1d_with_indices(input.native, options)
    TensorTuple(values = fromNative[D](native.get0()), indices = fromNative(native.get1))

  private def maxPool2dOptions[D <: FloatNN | Complex32](
      kernelSize: Int | (Int, Int),
      stride: Int | (Int, Int) | None.type,
      padding: Int | (Int, Int),
      dilation: Int | (Int, Int),
      ceilMode: Boolean
  ): MaxPool2dOptions = {
    val options: MaxPool2dOptions = MaxPool2dOptions(toNative(kernelSize))
    stride match
      case s: (Int | (Int, Int)) => options.stride().put(toNative(s))
      case None                  =>
    options.padding().put(toNative(padding))
    options.dilation().put(toNative(dilation))
    options.ceil_mode().put(ceilMode)
    options
  }

  /** Applies a 2D max pooling over an input signal composed of several input planes.
    *
    * @see
    *   [[torch.nn.MaxPool2d]] for details.
    * @param input
    *   input tensor $(\text{minibatch} , \text{in\_channels} , iH , iW)$, minibatch dim optional.
    * @param kernelSize
    *   size of the pooling region. Can be a single number or a tuple `(kH, kW)`
    * @param stride
    *   stride of the pooling operation. Can be a single number or a tuple `(sH, sW)`
    * @param padding
    *   $maxPoolPadding
    * @param dilation
    *   $dilation
    * @param ceilMode
    *   $ceilMode
    * @group nn_pooling
    */
  def max_pool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int) | None.type = None,
      padding: Int | (Int, Int) = 0,
      dilation: Int | (Int, Int) = 1,
      ceil_mode: Boolean = false
  ): Tensor[D] =
    val options: MaxPool2dOptions =
      maxPool2dOptions(kernel_size, stride, padding, dilation, ceil_mode)
    fromNative(torchNative.max_pool2d(input.native, options))

  /** Applies a 2D max pooling over an input signal composed of several input planes.
    *
    * @see
    *   [[torch.nn.MaxPool2d]] for details.
    * @param input
    *   input tensor $(\text{minibatch} , \text{in\_channels} , iH , iW)$, minibatch dim optional.
    * @param kernelSize
    *   size of the pooling region. Can be a single number or a tuple `(kH, kW)`
    * @param stride
    *   stride of the pooling operation. Can be a single number or a tuple `(sH, sW)`
    * @param padding
    *   $maxPoolPadding
    * @param dilation
    *   $dilation
    * @param ceilMode
    *   $ceilMode
    * @group nn_pooling
    */
  def max_pool2d_with_indices[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int) | None.type = None,
      padding: Int | (Int, Int) = 0,
      dilation: Int | (Int, Int) = 1,
      ceil_mode: Boolean = false
  ): TensorTuple[D] =
    val options: MaxPool2dOptions =
      maxPool2dOptions(kernel_size, stride, padding, dilation, ceil_mode)
    val native = torchNative.max_pool2d_with_indices(input.native, options)
    TensorTuple(values = fromNative[D](native.get0()), indices = fromNative(native.get1))

  private def maxPool3dOptions[D <: FloatNN | Complex32](
      kernel_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | None.type,
      padding: Int | (Int, Int, Int),
      dilation: Int | (Int, Int, Int),
      ceil_mode: Boolean
  ): MaxPool3dOptions = {
    val options: MaxPool3dOptions = MaxPool3dOptions(toNative(kernel_size))
    stride match
      case s: (Int | (Int, Int, Int)) => options.stride().put(toNative(s))
      case None                       =>
    options.padding().put(toNative(padding))
    options.dilation().put(toNative(dilation))
    options.ceil_mode().put(ceil_mode)
    options
  }

  /** Applies a 3D max pooling over an input signal composed of several input planes.
    *
    * @group nn_pooling
    */
  def max_pool3d[D <: Float32 | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | None.type = None,
      padding: Int | (Int, Int, Int) = 0,
      dilation: Int | (Int, Int, Int) = 1,
      ceil_mode: Boolean = false
  ): Tensor[D] =
    val options: MaxPool3dOptions =
      maxPool3dOptions(kernel_size, stride, padding, dilation, ceil_mode)
    fromNative(torchNative.max_pool3d(input.native, options))

  /** Applies a 3D max pooling over an input signal composed of several input planes.
    *
    * @group nn_pooling
    */
  def max_pool3d_with_indices[D <: Float16 | Float32 | Float64 | Complex32](
      input: Tensor[D],
      kernel_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int, Int) | None.type = None,
      padding: Int | (Int, Int, Int) = 0,
      dilation: Int | (Int, Int, Int) = 1,
      ceil_mode: Boolean = false
  ): TensorTuple[D] =
    val options: MaxPool3dOptions =
      maxPool3dOptions(kernel_size, stride, padding, dilation, ceil_mode)
    val native = torchNative.max_pool3d_with_indices(input.native, options)
    TensorTuple(values = fromNative[D](native.get0()), indices = fromNative(native.get1))

  // TODO max_unpool1d Computes a partial inverse of MaxPool1d.
  // TODO max_unpool2d Computes a partial inverse of MaxPool2d.
  // TODO max_unpool3d Computes a partial inverse of MaxPool3d.
  // TODO lp_pool1d Applies a 1D power-average pooling over an input signal composed of several input planes.
  // TODO lp_pool2d Applies a 2D power-average pooling over an input signal composed of several input planes.
  // TODO adaptive_max_pool1d Applies a 1D adaptive max pooling over an input signal composed of several input planes.
  // TODO adaptive_max_pool2d Applies a 2D adaptive max pooling over an input signal composed of several input planes.
  // TODO adaptive_max_pool3d Applies a 3D adaptive max pooling over an input signal composed of several input planes.
  // TODO adaptive_avg_pool1d Applies a 1D adaptive average pooling over an input signal composed of several input planes.
  // TODO adaptive_avg_pool2d Applies a 2D adaptive average pooling over an input signal composed of several input planes.
  // TODO adaptive_avg_pool3d Applies a 3D adaptive average pooling over an input signal composed of several input planes.
  // TODO fractional_max_pool2d Applies 2D fractional max pooling over an input signal composed of several input planes.
  // TODO fractional_max_pool3d Applies 3D fractional max pooling over an input signal composed of several input planes.
  def adaptive_avg_pool1d[D <: FloatNN | Complex32](
      input: Tensor[D],
      output_size: Long
  ): Tensor[D] =
    val options: AdaptiveAvgPool1dOptions = new AdaptiveAvgPool1dOptions(LongPointer(output_size))
    options.output_size().put(LongOptional(output_size))
    fromNative(torchNative.adaptive_avg_pool1d(input.native, options))

  def adaptive_avg_pool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      output_size: Long
  ): Tensor[D] =
    val options: AdaptiveAvgPool2dOptions = new AdaptiveAvgPool2dOptions(LongPointer(output_size))
    options.output_size().put(LongOptional(output_size))
    fromNative(torchNative.adaptive_avg_pool2d(input.native, options))

  def adaptive_avg_pool3d[D <: FloatNN | Complex32](
      input: Tensor[D],
      output_size: Long
  ): Tensor[D] =
    val options: AdaptiveAvgPool3dOptions = new AdaptiveAvgPool3dOptions(LongPointer(output_size))
    options.output_size().put(LongOptional(output_size))
    fromNative(torchNative.adaptive_avg_pool3d(input.native, options))

  def fractional_max_pool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernel_size: Long,
      output_ratio: Double,
      random_sample: Tensor[D],
      output_size: Long
  ): Tensor[D] =
    val options = new FractionalMaxPool2dOptions(LongOptional(kernel_size))
    options.output_size().put(LongExpandingArrayOptional(LongPointer(output_size)))
    options.output_ratio().put(DoubleExpandingArrayOptional(DoublePointer(output_ratio)))
    options._random_samples().put(random_sample.native)
    fromNative(torchNative.fractional_max_pool2d(input.native, options))

  def fractional_max_pool3d[D <: FloatNN | Complex32](
      input: Tensor[D],
      kernel_size: Long,
      output_ratio: Double,
      random_sample: Tensor[D],
      output_size: Long
  ): Tensor[D] =
    val options = new FractionalMaxPool3dOptions(LongOptional(kernel_size))
    options.output_size().put(LongExpandingArrayOptional(LongPointer(output_size)))
    options.output_ratio().put(DoubleExpandingArrayOptional(DoublePointer(output_ratio)))
    options._random_samples().put(random_sample.native)
    fromNative(torchNative.fractional_max_pool3d(input.native, options))

  def adaptive_max_pool1d[D <: FloatNN | Complex32](
      input: Tensor[D],
      output_size: Int | (Int, Int)
  ): Tensor[D] = {

    val options: AdaptiveMaxPool1dOptions =
      AdaptiveMaxPool1dOptions(toNative(output_size))
    fromNative(torchNative.adaptive_max_pool1d(input.native, options))
  }
  def adaptive_max_pool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      output_size: Int | (Int, Int, Int)
  ): Tensor[D] = {
    val options: AdaptiveMaxPool2dOptions =
      AdaptiveMaxPool2dOptions(toNative(output_size))
    fromNative(torchNative.adaptive_max_pool2d(input.native, options))
  }
  def adaptive_max_pool3d[D <: FloatNN | Complex32](
      input: Tensor[D],
      output_size: Int | (Int, Int) | (Int, Int, Int)
  ): Tensor[D] = {
    val options: AdaptiveMaxPool3dOptions =
      AdaptiveMaxPool3dOptions(toNative(output_size))
    fromNative(torchNative.adaptive_max_pool3d(input.native, options))
  }
  def lp_pool1d[D <: FloatNN | Complex32](
      input: Tensor[D],
      norm_type: Double,
      kernel_size: Int | (Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int), // None.type = None,
      padding: Int | (Int, Int) = 0,
      ceil_mode: Boolean = false
  ): Tensor[D] = {
    val options: LPPool1dOptions = LPPool1dOptions(norm_type, toNative(kernel_size))
    options.stride().put(toNative(stride))
    options.ceil_mode().put(ceil_mode)
    fromNative(torchNative.lp_pool1d(input.native, options))
  }
  def lp_pool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      norm_type: Double,
      kernel_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int), // None.type = None,
      padding: Int | (Int, Int, Int) = 0,
      ceil_mode: Boolean = false
  ): Tensor[D] = {
    val options: LPPool2dOptions = LPPool2dOptions(norm_type, toNative(kernel_size))
    options.stride().put(toNative(stride))
    options.ceil_mode().put(ceil_mode)
    fromNative(torchNative.lp_pool2d(input.native, options))
  }
  def lp_pool3d[D <: FloatNN | Complex32](
      input: Tensor[D],
      norm_type: Double,
      kernel_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int), // None.type = None,
      padding: Int | (Int, Int, Int) = 0,
      ceil_mode: Boolean = false
  ): Tensor[D] = {
    val options: LPPool3dOptions = LPPool3dOptions(norm_type, toNative(kernel_size))
    options.stride().put(toNative(stride))
    options.ceil_mode().put(ceil_mode)
    fromNative(torchNative.lp_pool3d(input.native, options))
  }
  //  def lPPool1dOptions[D <: FloatNN | Complex32](
  //      normType: Double,
  //      kernelSize: Int | (Int, Int),
  //      stride: Int | (Int, Int) | None.type,
  //      padding: Int | (Int, Int)
  //  ): LPPool1dOptions = {
  //    val options: LPPool1dOptions = LPPool1dOptions(normType, toNative(kernelSize))
  //    stride match
  //      case s: (Int | (Int, Int)) => options.stride().put(toNative(s))
  //      case None                   =>
  //    options.padding().put(toNative(padding))
  //    options
  //  }

  def max_unpool3d[D <: FloatNN | Complex32](
      input: Tensor[D],
      indices: Tensor[D],
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      output_size: Int | (Int, Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int), // None.type = None,
      padding: Int | (Int, Int, Int) = 0
  ): Tensor[D] = {
    val options: MaxUnpool3dFuncOptions = MaxUnpool3dFuncOptions(toNative(kernel_size))
    options.stride().put(toNative(stride))
    options.padding().put(toNative(padding))
    options.output_size().put(toNative(output_size))
    fromNative(torchNative.max_unpool3d(input.native, indices.native, options))
  }
  def max_unpool2d[D <: FloatNN | Complex32](
      input: Tensor[D],
      indices: Tensor[D],
      kernel_size: Int | (Int, Int),
      output_size: Int | (Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int), // None.type = None,
      padding: Int | (Int, Int) = 0
  ): Tensor[D] = {
    val options: MaxUnpool2dFuncOptions =
      MaxUnpool2dFuncOptions(toNative(kernel_size))
    options.stride().put(toNative(stride))
    options.padding().put(toNative(padding))
    options.output_size().put(toNative(output_size))
    fromNative(torchNative.max_unpool2d(input.native, indices.native, options))
  }
  def max_unpool1d[D <: FloatNN | Complex32](
      input: Tensor[D],
      indices: Tensor[D],
      kernel_size: Int | (Int, Int),
      output_size: Int | (Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int), // None.type = None,
      padding: Int | (Int, Int) = 0
  ): Tensor[D] = {
    val options: MaxUnpool1dFuncOptions =
      MaxUnpool1dFuncOptions(toNative(kernel_size))
    options.stride().put(toNative(stride))
    options.padding().put(toNative(output_size))
    fromNative(torchNative.max_unpool1d(input.native, indices.native, options))
  }
}
