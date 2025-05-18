//package torch.nn.modules.activation //SoftplusImpl
package torch
package nn
package modules
package activation

import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{ELUOptions, SoftplusImpl, SoftplusOptions, SoftshrinkOptions}
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
final class Softplus[D <: DType: Default](size: Int, threshold: Float, beta: Float)
    extends TensorModule[D]:

  val option = SoftplusOptions(size.toLong)
  option.beta().put(beta.toDouble)
  option.threshold().put(threshold.toDouble)
  override protected[torch] val nativeModule: SoftplusImpl = new SoftplusImpl(option)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()
  
  def apply(t: Tensor[D]): Tensor[D] = fromNative(nativeModule.forward(t.native))

  override def toString =
    getClass().getSimpleName() + s"(size=$size,threshold=$threshold,beta=$beta)"

object Softplus:
  def apply[D <: DType: Default](size: Int, threshold: Float, beta: Float): Softplus[D] = new Softplus(size, threshold, beta)