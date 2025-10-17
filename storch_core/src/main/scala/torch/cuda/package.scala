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

import org.bytedeco.pytorch.ProcessGroup
import org.bytedeco.pytorch.cuda.CUDAStream
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.global.torch_cuda as torchCuda
import torch.distribute.DistBackend

/** This package adds support for CUDA tensor types, that implement the same function as CPU
  * tensors, but they utilize GPUs for computation.
  */
package object cuda {

  /** Returns a Boolean indicating if CUDA is currently available. */
  def isAvailable: Boolean = torchNative.cuda_is_available()

  def is_available: Boolean = torchNative.cuda_is_available()

  def is_available(un_used: Int*): Boolean = torchNative.cuda_is_available()

  def device_count = torchNative.cuda_device_count()

  def cudnn_is_available: Boolean = torchNative.cudnn_is_available()
//
//  def getCurrentCUDASolverDnHandle = torchCuda.getCurrentCUDASolverDnHandle()
//
//  def getCudnnHandle = torchCuda.getCudnnHandle()

//  def empty_cache(un_used: Int*): Unit = torchCuda.empty_cache()
//
//  def memory_allocated(device: Device) = torchCuda.memory_allocated(device)
//
//  def memory_reserved(device: Device) = torchCuda.memory_reserved(device)

  def currentStreamCaptureStatusMayInitCtx = torchCuda.currentStreamCaptureStatusMayInitCtx()

  def manual_seed(seed: Long) = torchNative.cuda_manual_seed(seed)

  def manual_seed_all(seed: Long) = torchNative.cuda_manual_seed_all(seed)

  def cuda_manual_seed(seed: Long) = torchNative.cuda_manual_seed(seed)

  def cuda_manual_seed_all(seed: Long) = torchNative.cuda_manual_seed_all(seed)

  def cuda_synchronize = torchNative.cuda_synchronize()

  def cuda_synchronize(sync: Long) = torchNative.cuda_synchronize(sync)

  def CUDA_HELP = torchNative.CUDA_HELP()

  def getCUDAHooks = torchNative.getCUDAHooks()

  def set_device(device: Byte) = torchCuda.set_device(device)

  def device_synchronize = torchCuda.device_synchronize()

  def warn_or_error_on_sync = torchCuda.warn_or_error_on_sync()

  def getNumGPUs = torchCuda.getNumGPUs()

  def clearCublasWorkspaces = torchCuda.clearCublasWorkspaces()

  def getDefaultCUDAStream = torchCuda.getDefaultCUDAStream()

  def getCurrentCUDAStream = torchCuda.getCurrentCUDAStream()

  def setCurrentCUDAStream(stream: CUDAStream) = torchCuda.setCurrentCUDAStream(stream)

  def dataSize(dtype: Int) = torchCuda.dataSize(dtype)

  def getAllocator = torchCuda.getAllocator()

  def getMemoryFraction(frac: Byte) = torchCuda.getMemoryFraction(frac)

  def isEnabled: Boolean = torchCuda.isEnabled()

  def enable(flag: Boolean) = torchCuda.enable(flag)

  def contiguousIfZeroInStrides[D <: DType](tensor: Tensor[D]) = fromNative(
    torchCuda.contiguousIfZeroInStrides(tensor.native)
  )
}
