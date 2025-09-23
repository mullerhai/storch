package torch
package utils
package data
package sampler
package distribute

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  InputArchive,
  OutputArchive,
  SizeTOptional,
  SizeTVectorOptional,
  DistributedRandomSampler as DRS
}

class DistributedRandomSampler(
    size: Long,
    num_replicas: Long = 1,
    rank: Long = 0,
    allow_duplicates: Boolean = true
) extends DRS(size, num_replicas, rank, allow_duplicates)
    with Sampler {

  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def reset(): Unit = super.reset()

  override def next(batch_size: Long): SizeTVectorOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)

  override def index(): Long = super.index()
}
