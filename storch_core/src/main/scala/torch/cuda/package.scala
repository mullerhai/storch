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

  object distribute {
    def barrier[D <: DType](using backend: DistBackend[D]) = backend.barrier()

    def waitForPendingWorks[D <: DType](using backend: DistBackend[D]) = backend.waitForPendingWorks()

    def enableCollectivesTiming[D <: DType](using backend: DistBackend[D]) =
      backend.enableCollectivesTiming()

    def hasHooks[D <: DType](using backend: DistBackend[D]) = backend.hasHooks()

    def setGroupUid[D <: DType](uid: String)(using backend: DistBackend[D]) = backend.setGroupUid(uid)

    def getGroupUid[D <: DType](using backend: DistBackend[D]) = backend.getGroupUid()

    def getGroupDesc[D <: DType](using backend: DistBackend[D]) = backend.getGroupDesc()

    def getBoundDeviceId[D <: DType](using backend: DistBackend[D]) = backend.getBoundDeviceId()

    def reduce_scatter[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
                                                                                          backend: DistBackend[D]
    ) = backend.reduce_scatter(tensorSeq, tensorSeq2)

    def all_reduce[D <: DType](tensors: Seq[Tensor[D]])(using backend: DistBackend[D]) =
      backend.allreduce(tensors)

    def allreduce_sparse[D <: DType](tensors: Seq[Tensor[D]])(using backend: DistBackend[D]) =
      backend.allreduce_sparse(tensors)

    def allreduce_coalesced[D <: DType](tensors: Seq[Tensor[D]])(using backend: DistBackend[D]) =
      backend.allreduce_coalesced(tensors)

    def supportsSplitting[D <: DType](using backend: DistBackend[D]) = backend.supportsSplitting()

    def startCoalescing[D <: DType](using backend: DistBackend[D]) = backend.startCoalescing()

    def endCoalescing[D <: DType](using backend: DistBackend[D]) = backend.endCoalescing()

    def getBackendName[D <: DType](using backend: DistBackend[D]) = backend.getBackendName()

    def get_gpu_id[D <: DType](using backend: DistBackend[D]) = backend.getID()

    def get_world_size[D <: DType](using backend: DistBackend[D]) = backend.getSize()

    def all_gather[D <: DType](
                                tensorSeq: Seq[Tensor[D]],
                                tensorSeq2: Seq[Tensor[D]],
                                async_op: Boolean
                              )(using backend: DistBackend[D]) = backend.allgather(tensorSeq, tensorSeq2, async_op)

    def reduce[D <: DType](tensors: Seq[Tensor[D]])(using backend: DistBackend[D]) =
      backend.reduce(tensors)

    def send[D <: DType](tensorSeq: Seq[Tensor[D]], dstRank: Int, tag: Int)(using
                                                                            backend: DistBackend[D]
    ) = backend.send(tensorSeq, dstRank, tag)

    def recv[D <: DType](tensorSeq: Seq[Tensor[D]], srcRank: Int, tag: Int)(using
                                                                            backend: DistBackend[D]
    ) = backend.recv(tensorSeq, srcRank, tag)

    def recvAnySource[D <: DType](tensorSeq: Seq[Tensor[D]], tag: Int)(using
                                                                       backend: DistBackend[D]
    ) = backend.recvAnysource(tensorSeq, tag)

    def allgather_coalesced[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
                                                                                               backend: DistBackend[D]
    ) =
      backend.allgather_coalesced(tensorSeq, tensorSeq2)

    def allgather_into_tensor_coalesced[D <: DType](
                                                     tensorSeq: Seq[Tensor[D]],
                                                     tensorSeq2: Seq[Tensor[D]]
                                                   )(using backend: DistBackend[D]) =
      backend.allgather_into_tensor_coalesced(tensorSeq, tensorSeq2)

    def gather[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
                                                                                  backend: DistBackend[D]
    ) =
      backend.gather(tensorSeq, tensorSeq2)

    def scatter[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
                                                                                   backend: DistBackend[D]
    ) =
      backend.scatter(tensorSeq, tensorSeq2)

    def reduce_scatter_tensor_coalesced[D <: DType](
                                                     tensorSeq: Seq[Tensor[D]],
                                                     tensorSeq2: Seq[Tensor[D]]
                                                   )(using backend: DistBackend[D]) =
      backend.reduce_scatter_tensor_coalesced(tensorSeq, tensorSeq2)

    def all_to_all[D <: DType](tensorSeq: Seq[Tensor[D]], tensorSeq2: Seq[Tensor[D]])(using
                                                                                      backend: DistBackend[D]
    ) = backend.alltoall(tensorSeq, tensorSeq2)

    def get_rank[D <: DType](using backend: DistBackend[D]) = backend.getRank()

    def broadcast[D <: DType](tensorSeq: Seq[Tensor[D]])(using backend: DistBackend[D]) =
      backend.broadcast(tensorSeq)
  }


  /** Returns a Boolean indicating if CUDA is currently available. */
  def isAvailable: Boolean = torchNative.cuda_is_available()

  def device_count = torchNative.cuda_device_count()

//  def cudnn_is_available :Boolean = torchNative.cudnn_is_available()

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

//  def getCurrentCUDASolverDnHandle = torchCuda.getCurrentCUDASolverDnHandle()

  def getDefaultCUDAStream = torchCuda.getDefaultCUDAStream()

  def getCurrentCUDAStream = torchCuda.getCurrentCUDAStream()

  def setCurrentCUDAStream(stream: CUDAStream) = torchCuda.setCurrentCUDAStream(stream)

//  def getCudnnHandle = torchCuda.getCudnnHandle()

  def dataSize(dtype: Int) = torchCuda.dataSize(dtype)

  def getAllocator = torchCuda.getAllocator()

  def getMemoryFraction(frac: Byte) = torchCuda.getMemoryFraction(frac)

  def isEnabled: Boolean = torchCuda.isEnabled()

  def enable(flag: Boolean) = torchCuda.enable(flag)

//  def currentStreamCaptureStatusMayInitCtx = torchCuda.currentStreamCaptureStatusMayInitCtx()

  def contiguousIfZeroInStrides[D <: DType](tensor: Tensor[D]) = fromNative(
    torchCuda.contiguousIfZeroInStrides(tensor.native)
  )
}
