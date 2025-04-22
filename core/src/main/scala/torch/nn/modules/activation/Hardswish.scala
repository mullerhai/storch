//package torch.nn.modules.activation
package torch
package nn
package modules
package activation

import org.bytedeco.pytorch
import torch.internal.NativeConverters.fromNative
//import org.bytedeco.pytorch.{HardswishImpl, ELUOptions}
final class Hardswish[D <: DType: Default] extends TensorModule[D]:

  override def hasBias(): Boolean = false

  override def toString = getClass().getSimpleName()
  def apply(t: Tensor[D]): Tensor[D] = torch.hardswish(t)
