package torch
package ops

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.global.torch as TorchNative
import org.bytedeco.pytorch.{
  AutoFwGradMode,
  AutoGradMode,
  AutogradContext,
  AutogradMetaFactory,
  AutogradState,
  ForwardGrad,
  GradBucket,
  GradCallback,
  GradMode,
  LongArrayRefVector,
  NoGradGuard,
  PostAccumulateGradHook,
  SizeTVector,
  TensorOptional,
  TensorVector
}

trait GradOps {

  def gradCallback(): GradCallback = {
    // PostAccumulateGradHook
    val gradCallback = new GradCallback(new Pointer())
    gradCallback
  }
  def autoFwGradMode(enabled: Boolean = true): AutoFwGradMode = {
    val autoFwGradMode = new AutoFwGradMode(enabled)
    autoFwGradMode
  }

  def gradBucket[P <: DType](
      index: Long,
      bucketCount: Long,
      tensor: Tensor[P],
      offsets: SizeTVector,
      lengths: SizeTVector,
      sizesVec: LongArrayRefVector,
      parameters: TensorVector,
      sparseGradIndices: TensorOptional
  ): GradBucket = {
    val gradBucket = new GradBucket(
      index,
      bucketCount,
      tensor.native,
      offsets,
      lengths,
      sizesVec,
      parameters,
      sparseGradIndices
    )
    gradBucket
  }

  def autogradState(
      gradMode: Boolean,
      inferenceMode: Boolean,
      fwGradMode: Boolean,
      multiThreadingEnabled: Boolean
  ): AutogradState = {
    val auto = new AutogradState(gradMode, inferenceMode, fwGradMode, multiThreadingEnabled)
    auto
//    AutogradState.get_tls_state()
//    AutoFwGradMode
  }

  def forwardGrad(): ForwardGrad = {
    val forwardGrad = new ForwardGrad()
    forwardGrad
//    ForwardGrad.undef_grad()
  }
  
  def undef_grad() = {
    ForwardGrad.undef_grad()
  }
  
  def autogradContext(): AutogradContext = {
    val autogradContext = new AutogradContext()
    autogradContext
  }
  
  def require_grad(flag: Boolean) = TorchNative.requires_grad(flag)

  def no_grad_raw(): NoGradGuard = {
    val noGradGuard = new NoGradGuard()
    noGradGuard
  }

  def enable_grad(enable: Boolean = true): Unit = {
    GradMode.set_enabled(enable)
  }

  def disable_grad(disable: Boolean = true): Unit = {
    val flag = !disable
    GradMode.set_enabled(flag)
  }

  def is_enable(): Boolean = {
    GradMode.is_enabled()
  }

}
