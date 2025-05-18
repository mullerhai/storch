package torch
package nn
package modules
package normalization

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LayerNormImpl, LayerNormOptions, LongVector}
import internal.NativeConverters.fromNative

final class RMSNorm[ParamType <: FloatNN | ComplexNN: Default](
    normalizedShape: Seq[Int],
    eps: Double = 1e-05,
    elementWiseAffine: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")

  var weight: Tensor[ParamType] =
    registerParameter(torch.empty(normalizedShape, dtype = this.paramType), true, "weight")

  override def hasBias(): Boolean = false

  override def toString =
    s"${getClass.getSimpleName}(normalizedShape = ${normalizedShape.mkString(" ")}, elementWiseAffine = ${elementWiseAffine},eps=$eps )"

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val output =
      if elementWiseAffine then
        torch.rms_norm(
          input,
          normalizedShape.map(_.toLong),
          Some(weight.to(DType.float32)),
          Some(eps)
        )
      else torch.rms_norm(input, normalizedShape.map(_.toLong), None, Some(eps))
    output.to(input.dtype)
  }

object RMSNorm:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      normalized_shape: Seq[Int],
      eps: Double = 1e-05,
      element_wise_affine: Boolean = true
  ): RMSNorm[ParamType] =
    new RMSNorm(normalized_shape, eps, element_wise_affine)
