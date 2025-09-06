//package torch.nn.modules.activation //SoftplusImpl
package torch
package nn
package modules
package activation

import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{
  ELUOptions,
  SoftminImpl,
  SoftminOptions,
  SoftplusOptions,
  SoftshrinkOptions
}
//import torch.Tensor.fromNative
import torch.nn
import torch.nn.modules

/** Applies the Hyperbolic Tangent (Tanh) function element-wise. Tanh is defined as::
  *
  * TODO LaTeX
  *
  * Example:
  *
  * ```scala sc
  * import torch.*
  * val m = nn.Tanh()
  * val input = torch.randn(Seq(2))
  * val output = m(input)
  * ```
  */
final class Softmin[D <: DType: Default](dim: Int, threshold: Float, beta: Float)
    extends TensorModule[D]:

  val option = SoftminOptions(dim.toLong)

  override protected[torch] val nativeModule: SoftminImpl = new SoftminImpl(option)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))
  def forward(input: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(input.native))

  override def toString = getClass().getSimpleName() + s"(dim=$dim threshold=$threshold beta=$beta)"

object Softmin:
  def apply[D <: DType: Default](dim: Int, threshold: Float, beta: Float): Softmin[D] =
    new Softmin(dim, threshold, beta)
//  option.beta().put(beta) .toDouble
//  option.threshold().put(threshold) .toDouble
