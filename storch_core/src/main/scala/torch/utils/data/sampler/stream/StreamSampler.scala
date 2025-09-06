package torch
package utils
package data
package sampler
package stream

import torch.utils.data.dataset.java.SDataset
import torch.utils.data.sampler.BatchSizeSampler
import org.bytedeco.pytorch.StreamSampler as SS
class StreamSampler(epochSize: Long) extends SS(epochSize) with BatchSizeSampler
