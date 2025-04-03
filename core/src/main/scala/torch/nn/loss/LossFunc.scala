package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative

trait LossFunc extends Module {

  def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D]
}
