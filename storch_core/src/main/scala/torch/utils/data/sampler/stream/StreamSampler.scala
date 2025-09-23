package torch
package utils
package data
package sampler
package stream

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  DistributedSampler as DS,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  BatchSizeOptional,
  RandomSampler as RS
}
import torch.utils.data.sampler.BatchSizeSampler

import org.bytedeco.pytorch.StreamSampler as SS
class StreamSampler(epochSize: Long) extends SS(epochSize) with BatchSizeSampler {

  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def reset(): Unit = super.reset()

  override def next(batch_size: Long): BatchSizeOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}
