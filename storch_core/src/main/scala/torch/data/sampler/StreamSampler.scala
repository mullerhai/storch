package torch.data.sampler

import torch.data.dataset.java.SDataset
import torch.data.sampler.BatchSizeSampler
import org.bytedeco.pytorch.StreamSampler as SS
class StreamSampler(epochSize: Long) extends SS(epochSize) with BatchSizeSampler
