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
package ops

/** BLAS and LAPACK Operations
  *
  * https://pytorch.org/docs/stable/torch.html#blas-and-lapack-operations
  */
private[torch] trait BLASOps {
  def matmul[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    t1.matmul(t2)

  def dot[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    t1.dot(t2)
//    fromNative(
//    native.dot(other.native)
//  )

  // todo make sure the type of s is correct
  def vdot[D1 <: DType, D2 <: DType](t1: Tensor[D1], t2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    t1.vdot(t2)
//    fromNative(
//    native.vdot(other.native)
//  )
}
