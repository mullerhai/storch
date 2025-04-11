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
package functional

import torch.*
import torch.nn.functional as F
import torch.optim.Adam
import org.bytedeco.pytorch.OutputArchive
import scala.util.Random
import java.nio.file.Paths
import torch.Device.CUDA
import scala.util.Using
import org.bytedeco.javacpp.PointerScope
import torch.Device.CPU
import torch.nn.modules.HasParams
import Generators.genTensor

/*
 * dense = torch.randn(5, 5)
 * sparse = dense.to_sparse_csc()
 * sparse._nnz()
 *
 * dense = torch.zeros(3, 3, 1, 1)
 * dense[0, 0] = dense[1, 2] = dense[2, 1] = 1
 * dense.to_sparse_csc(dense_dim=2)
 *
*/
class sparseCooSuite extends munit.FunSuite {
  test("sparseCooSuite output shapes") {
    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    val batchSize =2
    val seqLength = 10
    val embedDim = 64
    val numHeads = 8
//    val multiheadAttention = nn.MultiheadAttention(embed_dim = 64,num_heads = 8,dropout = 0.1f)//,kdim = 8,bias = true,vdim = 8) //java.lang.RuntimeException: from is out of bounds for float
    val input = torch.randn(Seq(batchSize,seqLength,embedDim))
    val sparseTensor = input.to_sparse()
//    val out = multiheadAttention(input,input,input)
    println(s"sparseCooSuite ${sparseTensor.is_sparse()}||nnz ${sparseTensor.nnz()} |layout ${sparseTensor.layout} | ${sparseTensor.isSparse} |${sparseTensor.indices()}| value ${sparseTensor.values().shape.mkString(" ")}")

  }
}

/***
 * dense = torch.randn(5, 5)
 * sparse = dense.to_sparse_csc()
 * sparse._nnz()
 *
 * dense = torch.zeros(3, 3, 1, 1)
 * dense[0, 0] = dense[1, 2] = dense[2, 1] = 1
 * dense.to_sparse_csc(dense_dim=2)
 *
***/
class sparseCscSuite extends munit.FunSuite {
  test("sparseCscSuite output shapes") {
    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    val batchSize = 2
    val seqLength = 10
    val embedDim = 64
    val numHeads = 8
    //    val multiheadAttention = nn.MultiheadAttention(embed_dim = 64,num_heads = 8,dropout = 0.1f)//,kdim = 8,bias = true,vdim = 8) //java.lang.RuntimeException: from is out of bounds for float
    val input = torch.randn(Seq(batchSize, seqLength, embedDim))
    val sparseTensor = input.to_sparse_csc()
    //    val out = multiheadAttention(input,input,input)
    println(s"sparseCscSuite ${sparseTensor.is_sparse()} nnz ${sparseTensor.nnz()}||layout ${sparseTensor.layout}| ${sparseTensor.row_indices()} |${sparseTensor.ccol_indices()} | value ${sparseTensor.values().shape.mkString(" ")}")
    val denseTensor = sparseTensor.to_dense()
    println(s"denseTensor shape ${denseTensor.shape} layout ${denseTensor.layout} s_sparse ${denseTensor.is_sparse()}")

  }
}

/***
 *
 * crow_indices = [0, 2, 4]
 * col_indices = [0, 1, 0, 1]
 * values = [1, 2, 3, 4]
 * torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64),
 * torch.tensor(col_indices, dtype=torch.int64),
 * torch.tensor(values), dtype=torch.double)
 *
***/
class sparseCsrSuite extends munit.FunSuite {
  test("sparseCsrSuite output shapes") {
    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    val batchSize = 2
    val seqLength = 10
    val embedDim = 64
    val numHeads = 8
    //    val multiheadAttention = nn.MultiheadAttention(embed_dim = 64,num_heads = 8,dropout = 0.1f)//,kdim = 8,bias = true,vdim = 8) //java.lang.RuntimeException: from is out of bounds for float
    val input = torch.randn(Seq(batchSize, seqLength, embedDim))
    val sparseTensor = input.to_sparse_csr()
    //    val out = multiheadAttention(input,input,input)
    println(s"sparseCsrSuite ${sparseTensor.is_sparse()}|nnz ${sparseTensor.nnz()} |layout ${sparseTensor.layout} | ${sparseTensor.crow_indices()} | ${sparseTensor.col_indices()}| value ${sparseTensor.values().shape.mkString(" ")}")
    val denseTensor = sparseTensor.to_dense()
    println(s"denseTensor shape ${denseTensor.shape} layout ${denseTensor.layout} s_sparse ${denseTensor.is_sparse()}")

  }
}

/***
 * >>> ccol_indices = [0, 1, 2]
 * >>> row_indices = [0, 1]
 * >>> values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
 * >>> torch.sparse_bsc_tensor(torch.tensor(ccol_indices, dtype=torch.int64),
 * ...                         torch.tensor(row_indices, dtype=torch.int64),
 * ...                         torch.tensor(values), dtype=torch.double)
 * tensor(ccol_indices=tensor([0, 1, 2]),
 * row_indices=tensor([0, 1]),
 * values=tensor([[[1., 2.],
 *                        [3., 4.]],
 * [[5., 6.],
 *                        [7., 8.]]]), size=(2, 2), nnz=2, dtype=torch.float64,
 * layout=torch.sparse_bsc)
 *
***/
class sparseBscSuite extends munit.FunSuite {
  test("sparseBscSuite output shapes") {
    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    val batchSize = 2
    val seqLength = 10
    val embedDim = 64
    val numHeads = 8
    val blocksize =4
    //    val multiheadAttention = nn.MultiheadAttention(embed_dim = 64,num_heads = 8,dropout = 0.1f)//,kdim = 8,bias = true,vdim = 8) //java.lang.RuntimeException: from is out of bounds for float
    val input = torch.randn(Seq(batchSize, seqLength, embedDim))
    val sparseTensor = input.to_sparse_bsc(Seq(4))
    //    val out = multiheadAttention(input,input,input)
    println(s"sparseBscSuite ${sparseTensor.is_sparse()} |nnz ${sparseTensor.nnz()} |layout ${sparseTensor.layout} | ${sparseTensor.ccol_indices()} ${sparseTensor.row_indices() } | value ${sparseTensor.values().shape.mkString(" ")}")
    val denseTensor = sparseTensor.to_dense()
    println(s"denseTensor shape ${denseTensor.shape} layout ${denseTensor.layout} s_sparse ${denseTensor.is_sparse()}")

  }
}

/***
 *
 * crow_indices = [0, 1, 2]
 * col_indices = [0, 1]
 * values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
 * torch.sparse_bsr_tensor(torch.tensor(crow_indices, dtype=torch.int64),
 * torch.tensor(col_indices, dtype=torch.int64),
 * torch.tensor(values), dtype=torch.double)
***/
class sparseBsrSuite extends munit.FunSuite {
  test("sparseBsrSuite output shapes") {
    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    val batchSize = 2
    val seqLength = 10
    val embedDim = 64
    val numHeads = 8
    val blockSize = 4
    //    val multiheadAttention = nn.MultiheadAttention(embed_dim = 64,num_heads = 8,dropout = 0.1f)//,kdim = 8,bias = true,vdim = 8) //java.lang.RuntimeException: from is out of bounds for float
    val input = torch.randn(Seq(batchSize, seqLength, embedDim))
    val sparseTensor = input.to_sparse_bsr(Seq(4))
    //    val out = multiheadAttention(input,input,input)
    println(s"sparseBsrSuite ${sparseTensor.is_sparse()}|nnz ${sparseTensor.nnz()} |layout ${sparseTensor.layout}  | ${sparseTensor.crow_indices()} | ${sparseTensor.col_indices()} | value ${sparseTensor.values().shape.mkString(" ")}")
    val denseTensor = sparseTensor.to_dense()
    println(s"denseTensor shape ${denseTensor.shape} layout ${denseTensor.layout} s_sparse ${denseTensor.is_sparse()}")

  }
}

/***
 * compressed_indices = [0, 2, 4]
 * plain_indices = [0, 1, 0, 1]
 * values = [1, 2, 3, 4]
 * torch.sparse_compressed_tensor(torch.tensor(compressed_indices, dtype=torch.int64),
 * torch.tensor(plain_indices, dtype=torch.int64),
 * torch.tensor(values), dtype=torch.double, layout=torch.sparse_csr)
 *
 *
 * dense = torch.randn(5, 5)
 * sparse = dense.to_sparse_csc()
 * sparse._nnz()
  *
***/
class sparseCompressSuite extends munit.FunSuite {
  test("sparseCompressSuite output shapes") {
    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
    val batchSize = 2
    val seqLength = 10
    val embedDim = 64
    val numHeads = 8
    //    val multiheadAttention = nn.MultiheadAttention(embed_dim = 64,num_heads = 8,dropout = 0.1f)//,kdim = 8,bias = true,vdim = 8) //java.lang.RuntimeException: from is out of bounds for float
    val input = torch.randn(Seq(batchSize, seqLength, embedDim))
    val sparseTensor = input.to_sparse()
    //    val out = multiheadAttention(input,input,input)
    println(s"sparseComppressSuite ${sparseTensor.is_sparse()} |nnz ${sparseTensor.nnz()} |layout ${sparseTensor.layout}  | ${sparseTensor.indices()} | value ${sparseTensor.values().shape.mkString(" ")}")

    val denseTensor = sparseTensor.to_dense()
    println(s"denseTensor shape ${denseTensor.shape} layout ${denseTensor.layout} s_sparse ${denseTensor.is_sparse()}")

  }
}

//class LNet[D <: BFloat16 | Float32 : Default] extends TensorModule[D] {
//
//  val conv1 = register(nn.Conv2d(1, 6, 5))
//  val conv2 = register(nn.Conv2d(6, 16, 5))
//  val fc1 = register(nn.Linear(16 * 4 * 4, 120))
//  val fc2 = register(nn.Linear(120, 84))
//  val fc3 = register(nn.Linear(84, 10))
//
//  def apply(i: Tensor[D]): Tensor[D] =
//    var x = F.maxPool2d(F.relu(conv1(i)), (2, 2))
//    x = F.maxPool2d(F.relu(conv2(x)), 2)
//    x = x.view(-1, 16 * 4 * 4) // all dimensions except the batch dimension
//    x = F.relu(fc1(x))
//    x = F.relu(fc2(x))
//    x = fc3(x)
//    x
//}

//class ModelCompileSuite extends munit.FunSuite {
//  test("compileSuite ") {
//    //    val m1 = nn.TransformerEncoderLayer(d_model = 8,n_head = 4)
//    val model = LNet()
//    val optModel = torch.compile(model)
//
//    //    val out = multiheadAttention(input,input,input)
//    println(s"modelSuite ${optModel}")
//  }
//}


class SparseSuite extends TensorCheckSuite {

  // TODO Test multi-dimensional tensors
  testUnaryOp(
    op = nn.functional.oneHot(_, numClasses = 6),
    opName = "nn.functional.oneHot",
    inputTensor = Tensor(3L),
    expectedTensor = Tensor(Seq(0L, 0L, 0L, 1L, 0L, 0L)),
    // TODO Fix genTensor for cases where the tensor type is not a union, but a concrete one, such as Tensor[Int64]
    genTensor = genTensor[Int64](filterDTypes = true)
  )
}
