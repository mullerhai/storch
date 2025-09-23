package torch
package utils
package trainer

import torch.nn.loss.CrossEntropyLoss
import torch.optim.AdamW
import org.bytedeco.javacpp.PointerScope
import scala.util.Using
import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.loss.LossFunc
import torch.optim.Optimizer
import torch.utils.data.{Dataset, DataLoader}

/** LSTM模型训练器，继承自Trainable trait 使用AdamW优化器和CrossEntropyLoss损失函数
  */
class LstmTrainer[D <: BFloat16 | FloatNN: Default](
    model: LstmNet[D],
    train_loader: DataLoader[D],
    eval_loader: DataLoader[D],
    criterion: CrossEntropyLoss = new CrossEntropyLoss(),
    optimizer_type: String = "adamW",
    device: Device = Device.CPU
) extends Trainable[D, LstmNet[D]](
      model,
      train_loader,
      eval_loader,
      optimizer_type = "adamW",
      criterion = new CrossEntropyLoss(),
      device
    ) {

  /** 实现训练方法
    */
  override def train(
      num_epochs: Int,
      learning_rate: Float = 0.001f,
      batch_size: Int,
      eval_interval: Int = 200
  ): LstmNet[D] = {
    // 创建AdamW优化器
    val optimizer = optimizer_type.toLowerCase match {
      case "adamw"   => torch.optim.AdamW(model.parameters, learning_rate)
      case "adam"    => torch.optim.Adam(model.parameters, learning_rate)
      case "sgd"     => torch.optim.SGD(model.parameters, learning_rate)
      case "rmsprop" => torch.optim.RMSprop(model.parameters, learning_rate)
//          case "lbfgs" => torch.optim.LBFGS(model.parameters, learning_rate)
      case "adagrad" => torch.optim.Adagrad(model.parameters, learning_rate)
      case _         => throw new IllegalArgumentException(s"不支持的优化器类型: $optimizer_type")
    }
//    val optimizer = AdamW(model.parameters(true), learning_rate)

    model.to(device)
//    val criterion = new CrossEntropyLoss()
    // 按照注释要求使用PointerScope
    Using.resource(new PointerScope()) { _ =>
      for (epoch <- 1 to num_epochs) {
        model.train()
        var totalLoss = 0.0f
        val exampleIter = train_loader.iterator
        var batchIndex = 0
        for ((inputs, targets) <- train_loader) {
          // 将数据移到目标设备
          val inputsDevice = inputs.to(device)
          val targetsDevice = targets.to(device)

          // 前向传播
          val outputs = model(inputsDevice)
          val loss = criterion(outputs, targetsDevice)

          // 反向传播和优化
          optimizer.zeroGrad()
          loss.backward()
          optimizer.step()

          totalLoss = totalLoss + loss.item.asInstanceOf[Float]
          batchIndex = batchIndex + 1
          // 定期评估
          if (batchIndex % eval_interval == 0) {
            val (evalLoss, accuracy) = evaluate()
            println(
              s"Epoch: $epoch, Iteration: ${batchIndex}, | Training loss: ${loss.item}%.4f " +
                s"Train Loss: ${totalLoss / eval_interval}, Eval Loss: $evalLoss, Accuracy: $accuracy"
            )
            totalLoss = 0.0f
            model.train()
          }
        }

        // 每个epoch结束后评估
        val (evalLoss, accuracy) = evaluate()
        println(s"Epoch $epoch completed. Eval Loss: $evalLoss, Accuracy: $accuracy")
      }
    }

    model
  }

  /** 实现评估方法，按照注释要求使用torch.noGrad()
    */
  override def evaluate(): (Float, Float) = {
    model.eval()
    var totalLoss = 0.0f
    var correct = 0L
    var total = 0L

    // 按照注释要求使用torch.noGrad()
    torch.noGrad {
      for ((inputs, targets) <- eval_loader) {
        val inputsDevice = inputs.to(device)
        val targetsDevice = targets.to(device)

        val outputs = model(inputsDevice)
        val loss = criterion(outputs, targetsDevice)

//        val rec: Float = loss.item
        totalLoss = totalLoss + loss.item.asInstanceOf[Float]

        // 计算准确率
        val predictions = outputs.argmax(dim = 1)
        correct += predictions.eq(targetsDevice).sum().item
        total += targetsDevice.numel
      }
    }

    val avgLoss = totalLoss / eval_loader.size
    val accuracy = correct.toFloat / total.toFloat
    (avgLoss, accuracy)
  }
}
