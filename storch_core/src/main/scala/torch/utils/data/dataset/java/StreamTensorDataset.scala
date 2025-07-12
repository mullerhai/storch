package torch.utils.data.dataset.java

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{InputArchive, OutputArchive, SizeTOptional, SizeTVectorOptional, T_TensorT_TensorTensor_T_T, T_TensorTensor_T, T_TensorTensor_TOptional, TensorExampleVector, TensorMapper, TensorVector, TransformerImpl, TransformerOptions, kCircular, kGELU, kReflect, kReplicate, kZeros, ChunkBatchDataset as CBD, JavaStreamTensorDataset as STD, RandomSampler as RS, SequentialSampler as SS}
import torch.utils.data.dataset.Dataset
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.utils.data.datareader.TensorExampleVectorReader
import torch.utils.data.datareader
class StreamTensorDataset(reader: datareader.TensorExampleVectorReader) extends STD {

  override def get_batch(request: Long): TensorExampleVector = reader.tensorExampleVec

  override def size(): SizeTOptional = new SizeTOptional(reader.tensorExampleVec.size)
}
