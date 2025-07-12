package torch.utils.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ExampleVectorOptional, InputArchive, OutputArchive, SizeTOptional, SizeTVectorOptional, T_TensorT_TensorTensor_T_T, T_TensorTensor_T, T_TensorTensor_TOptional, TensorMapper, TensorVector, TransformerImpl, TransformerOptions, kCircular, kGELU, kReflect, kReplicate, kZeros, ChunkBatchDataset as CBD, JavaStatefulBatchDataset as SBD, RandomSampler as RS, SequentialSampler as SS}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.datareader.ExampleVectorReader
import torch.utils.data.datareader

class StatefulBatchDataset(reader: datareader.ExampleVectorReader) extends SBD(reader) {

  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
