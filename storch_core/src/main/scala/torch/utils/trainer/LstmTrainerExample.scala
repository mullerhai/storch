package torch
package utils
package trainer

import torch.utils.data.{Dataset, DataLoader}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.loss.LossFunc
import torch.optim.Optimizer
import torch.utils.data.{Dataset, DataLoader}
import torch.*
import torch.nn as nn

object LstmTrainerExample {
  def main(args: Array[String]): Unit = {
    // 定义模型参数
    val inputSize = 28
    val hiddenSize = 128
    val numLayers = 2
    val numClasses = 10
    val batchSize = 64
    val numEpochs = 10
    val learningRate = 0.001f

    // 创建LSTM模型
    val model = new LstmNet[Float32](inputSize, hiddenSize, numLayers, numClasses)

    // 假设已经有训练和评估数据加载器
//     val trainDataset: Dataset = new ExampleDataset(trainExamples)
//     val evalDataset: Dataset = new ExampleDataset(evalExamples)
//     val trainLoader = DataLoader(trainDataset, batchSize, shuffle = true)
//     val evalLoader = DataLoader(evalDataset, batchSize, shuffle = false)
//
//
//     val trainer = new LstmTrainer(model, trainLoader, evalLoader, Device.CUDA if torch.cuda.isAvailable else Device.CPU)
//
//     开始训练
//     trainer.train(numEpochs, learningRate, batchSize)
  }
}
