package torch
package utils
package data
package dataloader


import org.bytedeco.pytorch.{ChunkDataset, ChunkDatasetOptions, ChunkMapDataset, ChunkRandomDataLoader, ChunkSharedBatchDataset, Example, ExampleIterator, ExampleStack, ExampleVector}
import torch.utils.data.{DataLoaderOptions}
import torch.{DType, Default}
import torch.utils.data.sampler.Sampler

import scala.collection.mutable.ArrayBuffer
import torch.utils.data.dataloader.DataLoaderOptions
trait DataLoader






case class DataLoaderOptions(batch_size: Int = 1, shuffle: Boolean = true, sampler: Sampler = null,
                             batch_sampler: Sampler = null, num_workers: Int = 0, collate_fn: Any = null,
                             pin_memory: Boolean = false, drop_last: Boolean = false, timeout: Int = 0,
                             worker_init_fn: Any = null, prefetch_factor: Int = 2,
                             persistent_workers: Boolean = false)


case class TensorDataLoaderOptions(batch_size: Int = 1, shuffle: Boolean = false, sampler: Sampler = null,
                                   batch_sampler: Sampler = null, num_workers: Int = 0, collate_fn: Any = null,
                                   pin_memory: Boolean = false, drop_last: Boolean = false, timeout: Int = 0,
                                   worker_init_fn: Any = null, prefetch_factor: Int = 2,
                                   persistent_workers: Boolean = false)