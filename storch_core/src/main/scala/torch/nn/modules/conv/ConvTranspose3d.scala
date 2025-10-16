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
package modules
package conv

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ConvTranspose3dImpl,
  LongArrayRefOptional,
  ConvTranspose3dOptions,
  kZeros,
  kReflect,
  kReplicate,
  kCircular
}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.conv.ConvTranspose3d.PaddingMode

/** Applies a 2D convolution over an input signal composed of several input planes.
  *
  * @group nn_conv
  */
final class ConvTranspose3d[ParamType <: FloatNN | ComplexNN: Default](
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int | (Int, Int) | (Int, Int, Int),
    stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 1,
    padding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 0,
    outputPadding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 0,
    groups: Int | Option[Int] = 1,
    bias: Boolean | Option[Boolean] = true,
    dilation: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
      Option[(Int, Int, Int)] = 1,
    paddingMode: PaddingMode | String | Option[String] = PaddingMode.Zeros
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  private val options =
    new ConvTranspose3dOptions(inChannels.toLong, outChannels.toLong, toNative(kernelSize))
  stride match {
    case s: Int             => options.stride().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int)      => options.stride().put(Array(s._1.toLong, s._2.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) => options.stride().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case s: Option[Int] =>
      if s.isDefined then options.stride().put(Array(s.get.toLong, s.get.toLong, s.get.toLong)*)
    case s: Option[(Int, Int)] =>
      if s.isDefined then
        options.stride().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._2.toLong)*)
    case s: Option[(Int, Int, Int)] =>
      if s.isDefined then
        options.stride().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._3.toLong)*)
  }
  padding match {
    case s: Int             => options.padding().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int)      => options.padding().put(Array(s._1.toLong, s._2.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) => options.padding().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case s: Option[Int] =>
      if s.isDefined then options.padding().put(Array(s.get.toLong, s.get.toLong, s.get.toLong)*)
    case s: Option[(Int, Int)] =>
      if s.isDefined then
        options.padding().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._2.toLong)*)
    case s: Option[(Int, Int, Int)] =>
      if s.isDefined then
        options.padding().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._3.toLong)*)

  }
  outputPadding match {
    case s: Int => options.output_padding().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int) =>
      options.output_padding().put(Array(s._1.toLong, s._2.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) =>
      options.output_padding().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case s: Option[Int] =>
      if s.isDefined then
        options.output_padding().put(Array(s.get.toLong, s.get.toLong, s.get.toLong)*)
    case s: Option[(Int, Int)] =>
      if s.isDefined then
        options.output_padding().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._2.toLong)*)
    case s: Option[(Int, Int, Int)] =>
      if s.isDefined then
        options.output_padding().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._3.toLong)*)
  }
  dilation match {
    case s: Int             => options.dilation().put(Array(s.toLong, s.toLong, s.toLong)*)
    case s: (Int, Int)      => options.dilation().put(Array(s._1.toLong, s._2.toLong, s._2.toLong)*)
    case s: (Int, Int, Int) => options.dilation().put(Array(s._1.toLong, s._2.toLong, s._3.toLong)*)
    case s: Option[Int] =>
      if s.isDefined then options.dilation().put(Array(s.get.toLong, s.get.toLong, s.get.toLong)*)
    case s: Option[(Int, Int)] =>
      if s.isDefined then
        options.dilation().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._2.toLong)*)
    case s: Option[(Int, Int, Int)] =>
      if s.isDefined then
        options.dilation().put(Array(s.get._1.toLong, s.get._2.toLong, s.get._3.toLong)*)
  }
  groups match {
    case g: Int         => options.groups().put(g)
    case g: Option[Int] => if g.isDefined then options.groups().put(g.get)
  }
  bias match {
    case b: Boolean         => options.bias().put(b)
    case b: Option[Boolean] => if b.isDefined then options.bias().put(b.get)
  }

  paddingMode match
    case PaddingMode.Zeros | "zeros" | "Zeros" | Some("zeros") | Some("Zeros") =>
      options.padding_mode().put(new kZeros)
    case PaddingMode.Reflect | "reflect" | "Reflect" | Some("reflect") | Some("Reflect") =>
      options.padding_mode().put(new kReflect)
    case PaddingMode.Replicate | "replicate" | "Replicate" | Some("replicate") | Some(
          "Replicate"
        ) =>
      options.padding_mode().put(new kReplicate)
    case PaddingMode.Circular | "circular" | "Circular" | Some("cirular") | Some("Cirular") =>
      options.padding_mode().put(new kCircular)

  override private[torch] val nativeModule: ConvTranspose3dImpl = ConvTranspose3dImpl(options)
  nativeModule.to(paramType.toScalarType, false)

  def apply(t: Tensor[ParamType]): Tensor[ParamType] = fromNative(nativeModule.forward(t.native))

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = fromNative(
    nativeModule.forward(input.native)
  )

  def apply(input: Tensor[ParamType], output_size: Seq[Int]): Tensor[ParamType] = {

    val outputSize = LongArrayRefOptional(output_size.map(_.toLong)*)
    fromNative(nativeModule.forward(input.native, outputSize))
  }
  def forward(input: Tensor[ParamType], output_size: Seq[Int]): Tensor[ParamType] = {

    val outputSize = LongArrayRefOptional(output_size.map(_.toLong)*)
    fromNative(nativeModule.forward(input.native, outputSize))
  }

  def weight: Tensor[ParamType] = fromNative(nativeModule.weight)

  def bias_(): Tensor[ParamType] = fromNative(nativeModule.bias)

  def bias(un_used: Int*): Tensor[ParamType] = fromNative(nativeModule.bias)

  def reset(): Unit = nativeModule.reset()

  def reset_parameters(): Unit = nativeModule.reset_parameters()

  override def hasBias(): Boolean = options.bias().get()

  override def toString =
    s"${getClass.getSimpleName} ($inChannels, $outChannels, kernelSize=$kernelSize, stride=$stride, padding=$padding, bias=$bias)"

object ConvTranspose3d:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      in_channels: Int,
      out_channels: Int,
      kernel_size: Int | (Int, Int) | (Int, Int, Int),
      stride: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 1,
      padding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 0,
      output_padding: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 0,
      groups: Int | Option[Int] = 1,
      bias: Boolean | Option[Boolean] = true,
      dilation: Int | (Int, Int) | (Int, Int, Int) | Option[Int] | Option[(Int, Int)] |
        Option[(Int, Int, Int)] = 1,
      padding_mode: PaddingMode | String | Option[String] = PaddingMode.Zeros
  ): ConvTranspose3d[ParamType] = new ConvTranspose3d(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    output_padding,
    groups,
    bias,
    dilation,
    padding_mode
  )
  enum PaddingMode:
    case Zeros, Reflect, Replicate, Circular
