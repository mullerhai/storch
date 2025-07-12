package torch
package utils
package tensorboard

import com.google.protobuf.ByteString
import org.tensorflow.framework.histogram.HistogramProto
import org.tensorflow.framework.summary.Summary.{Audio, Image}
import org.tensorflow.framework.summary.SummaryMetadata.PluginData
import org.tensorflow.framework.summary.{DataClass, Summary, SummaryMetadata}
import org.tensorflow.framework.tensor.TensorProto
import org.tensorflow.framework.tensor_shape.TensorShapeProto
import org.tensorflow.framework.types.DataType
import org.tensorflow.util.event.LogMessage.Level
import org.tensorflow.util.event.LogMessage.Level.INFO
import org.tensorflow.util.event.SessionLog.SessionStatus
import org.tensorflow.util.event.*

import java.io.{DataOutputStream, FileOutputStream}
import java.nio.ByteBuffer
import java.util.zip.CRC32
import scala.collection.mutable.ListBuffer


class SummaryWriter(logDir: String,tfEventFilePath:String = "train.tfevents") {
  private val logFilePath = s"$logDir/${tfEventFilePath}"
  private val fileOutputStream = new FileOutputStream(logFilePath)
  private val dataOutputStream = new DataOutputStream(fileOutputStream)
  private val writer = new TFEventWriter(dataOutputStream)

  writeFileVersionEvent("brain.Event:2")
//  writeSessionLogEvent("brain.Event:2")
  
  def addScalar(tag: String, scalarValue: Double, globalStep: Long): Unit = {
//    val tensorProto = TensorProto(
//      dtype = DataType.DT_FLOAT,
//      tensorShape = Some(TensorShapeProto()),
//      floatVal = Seq(scalarValue.toFloat)
//    )

    val simpleValue = Summary.Value.Value.SimpleValue(
      value = scalarValue.toFloat
    )

    val pluginData = PluginData(
      pluginName = "scalars"
    )

    val metadata = SummaryMetadata(
      pluginData = Some(pluginData),
      dataClass = DataClass.DATA_CLASS_SCALAR
    )

    val summaryValue = Summary.Value(
      tag = tag,
      metadata = Some(metadata),
      value = Summary.Value.Value.SimpleValue(scalarValue.toFloat)
    )

    val summary = Summary(
      value = Seq(summaryValue)
    )

    writeEvent(summary, globalStep)
  }

  def addScalars(mainTag: String, tagScalarsDict: Map[String, Double], globalStep: Long): Unit = {
    val values = tagScalarsDict.map { case (tag, value) =>
      val tensorProto = TensorProto(
        dtype = DataType.DT_FLOAT,
        tensorShape = Some(TensorShapeProto()),
        floatVal = Seq(value.toFloat)
      )

      val pluginData = PluginData(
        pluginName = "scalars"
      )

      val metadata = SummaryMetadata(
        pluginData = Some(pluginData),
        dataClass = DataClass.DATA_CLASS_SCALAR
      )

      Summary.Value(
        tag = s"$mainTag/$tag",
        metadata = Some(metadata),
        value = Summary.Value.Value.Tensor(tensorProto)
      )
    }.toSeq

    val summary = Summary(
      value = values
    )

    writeEvent(summary, globalStep)
  }

  def addTensors(tag: String, tensors: Seq[Double], globalStep: Long): Unit = {
    val tensorProto = TensorProto(
      dtype = DataType.DT_FLOAT,
      tensorShape = Some(TensorShapeProto(dim = tensors.length :: Nil map (d => TensorShapeProto.Dim(size = d.toLong)))),
      floatVal = tensors.map(_.toFloat)
    )

    val pluginData = PluginData(
      pluginName = "tensors"
    )

    val metadata = SummaryMetadata(
      pluginData = Some(pluginData),
      dataClass = DataClass.DATA_CLASS_TENSOR
    )

    val summaryValue = Summary.Value(
      tag = tag,
      metadata = Some(metadata),
      value = Summary.Value.Value.Tensor(tensorProto)
    )

    val summary = Summary(
      value = Seq(summaryValue)
    )

    writeEvent(summary, globalStep)
  }

  def addAudio(tag: String, audioData: Array[Byte], sampleRate: Float, globalStep: Long): Unit = {
    val audioByteString = ByteString.copyFrom(audioData)
    val audioSummary = Audio(
      sampleRate = sampleRate,
      numChannels = 1,
      lengthFrames = audioData.length,
      encodedAudioString = audioByteString
    )

    val pluginData = PluginData(
      pluginName = "audio"

    )

    val metadata = SummaryMetadata(
      pluginData = Some(pluginData),
      dataClass = DataClass.DATA_CLASS_BLOB_SEQUENCE
    )

    val summaryValue = Summary.Value(
      tag = tag,
      metadata = Some(metadata),
      value = Summary.Value.Value.Audio(audioSummary)
    )

    val summary = Summary(
      value = Seq(summaryValue)
    )

    writeEvent(summary, globalStep)
  }

  def addImage(tag: String, imageData: Array[Byte], width: Int, height: Int, globalStep: Long): Unit = {
    val imageByteString = ByteString.copyFrom(imageData)
    val imageSummary = Image(
      colorspace = 3,
      height = height,
      width = width,
      encodedImageString = imageByteString
    )

    val pluginData = PluginData(
      pluginName = "images"

    )

    val metadata = SummaryMetadata(
      pluginData = Some(pluginData),
      dataClass = DataClass.DATA_CLASS_BLOB_SEQUENCE
    )

    val summaryValue = Summary.Value(
      tag = tag,
      metadata = Some(metadata),
      value = Summary.Value.Value.Image(imageSummary)
    )

    val summary = Summary(
      value = Seq(summaryValue)
    )

    writeEvent(summary, globalStep)
  }

  def addHistogram(tag: String, values: Seq[Double], globalStep: Long): Unit = {
    val minVal = values.min
    val maxVal = values.max
    val num = values.length.toLong
    val sum = values.sum
    val sumSquares = values.map(x => x * x).sum

    val histProto = HistogramProto(
      min = minVal,
      max = maxVal,
      num = num,
      sum = sum,
      sumSquares = sumSquares,
      bucket = ListBuffer.fill(values.length)(1.0).toSeq,
      bucketLimit = values.sorted
    )

    val pluginData = PluginData(
      pluginName = "histograms"
//      dataClass = DataClass.DATA_CLASS_UNKNOWN
    )

    val metadata = SummaryMetadata(
      pluginData = Some(pluginData),
      dataClass = DataClass.DATA_CLASS_BLOB_SEQUENCE
    )

    val summaryValue = Summary.Value(
      tag = tag,
      metadata = Some(metadata),
      value = Summary.Value.Value.Histo(histProto)
    )

    val summary = Summary(
      value = Seq(summaryValue)
    )

    writeEvent(summary, globalStep)
  }

  def writeFileVersionEvent(tag: String, fileVersion: String = "brain.Event:2"): Unit = {
    val sourceMetadata = SourceMetadata(
      writer = "tensorboard.summary.writer.event_file_writer" //"tensorflow.core.util.events_writer"
    )
    // 创建 Event 消息，添加 fileVersion 字段  Event.What.Summary(summary)
    val eventMeta = Event(
      what = Event.What.FileVersion(fileVersion),
      sourceMetadata = Some(sourceMetadata)
    )
    writer.write(eventMeta.toByteArray)
  }

  def writeSessionLogEvent(tag: String, sessionStatus: SessionStatus = SessionStatus.START): Unit = {
    val sourceMetadata = SourceMetadata(
      writer = "tensorboard.summary.writer.event_file_writer" //"tensorflow.core.util.events_writer"
    )
    // 创建 Event 消息，添加 fileVersion 字段  Event.What.Summary(summary)
    val eventSession = Event(
      what = Event.What.SessionLog(SessionLog(status = SessionStatus.START)),
      sourceMetadata = Some(sourceMetadata)
    )
    writer.write(eventSession.toByteArray)
  }

  def writeLogMessageEvent(message: String, logLevel: Level = INFO): Unit = {
    val sourceMetadata = SourceMetadata(
      writer = "tensorboard.summary.writer.event_file_writer" //"tensorflow.core.util.events_writer"
    )
    // 创建 Event 消息，添加 fileVersion 字段  Event.What.Summary(summary)
    val eventLogger = Event(
      what = Event.What.LogMessage(LogMessage(level = logLevel, message = message)),
      sourceMetadata = Some(sourceMetadata)
    )
    writer.write(eventLogger.toByteArray)
  }

  def writeTaggedRunMetadataEvent(tag: String, runMetadata: ByteString): Unit = {
    val sourceMetadata = SourceMetadata(
      writer = "tensorboard.summary.writer.event_file_writer" //"tensorflow.core.util.events_writer"
    )
    // 创建 Event 消息，添加 fileVersion 字段  Event.What.Summary(summary)
    val eventLogger = Event(
      what = Event.What.TaggedRunMetadata(TaggedRunMetadata(tag = tag, runMetadata = runMetadata)),
      sourceMetadata = Some(sourceMetadata)
    )
    writer.write(eventLogger.toByteArray)
  }

  private def writeEvent(summary: Summary, globalStep: Long): Unit = {
    val event = Event(
      wallTime = System.currentTimeMillis() / 1000.0,
      step = globalStep,
      what = Event.What.Summary(summary),
//      fileVersion = Some("brain.Event:2")
    )

    val serializedEvent = event.toByteArray
    writer.write(serializedEvent)
//    writeTFRecord(serializedEvent)
  }

  private def writeTFRecord(data: Array[Byte]): Unit = {
    val length = data.length.toLong
    val lengthBytes = ByteBuffer.allocate(8).putLong(length).array()
    dataOutputStream.write(lengthBytes)

    val lengthCrc = calculateCrc(lengthBytes)
    dataOutputStream.writeInt(lengthCrc)

    dataOutputStream.write(data)

    val dataCrc = calculateCrc(data)
    dataOutputStream.writeInt(dataCrc)
  }

  private def calculateCrc(data: Array[Byte]): Int = {
    val crc32 = new CRC32()
    crc32.update(data)
    (crc32.getValue & 0xFFFFFFFFL).toInt
  }

  def close(): Unit = {
    println("SummaryWriter closed.")
    dataOutputStream.close()
    fileOutputStream.close()
  }
}
