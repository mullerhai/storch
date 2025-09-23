package torch.utils.data.sampler

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Sampler as SM,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  SizeTVectorOptional,
  RandomSampler as RS
}

trait Sampler extends SM {

//  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)
//
//  override def next(batch_size: Long): SizeTVectorOptional = super.next(batch_size)
//
//  override def save(archive: OutputArchive): Unit = super.save(archive)
//
//  override def load(archive: InputArchive): Unit = super.load(archive)
}
