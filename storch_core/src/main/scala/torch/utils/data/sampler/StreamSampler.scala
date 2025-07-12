package torch.utils.data.sampler

import torch.utils.data.dataset.java.SDataset
import torch.utils.data.sampler.BatchSizeSampler
import org.bytedeco.pytorch.StreamSampler as SS
class StreamSampler(epochSize: Long) extends SS(epochSize) with BatchSizeSampler
