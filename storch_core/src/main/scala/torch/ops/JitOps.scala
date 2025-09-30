package torch
package ops

import org.bytedeco.javacpp.{BytePointer, Pointer, PointerPointer}
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.{Tensor as PytorchTensor, *}
import torch.nn.modules.TensorModule

import java.io.{ByteArrayOutputStream, InputStream, ObjectOutputStream}
import java.nio.ByteBuffer
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import torch.internal.NativeConverters.{fromNative, toScalar}

trait JitOps {

  def compile[ParamType <: FloatNN | ComplexNN: Default](
      model: TensorModule[ParamType]
  ): CompilationUnit =
    val bytes = modelToBytes(model)
    val bytePointer = new BytePointer(ByteBuffer.wrap(bytes))
    torchNative.compile(bytePointer)

  def modelToBytes[ParamType <: FloatNN | ComplexNN: Default](
      model: TensorModule[ParamType]
  ): Array[Byte] = {
    val byteStream = new ByteArrayOutputStream()
    val objectStream = new ObjectOutputStream(byteStream)
    try {
      objectStream.writeObject(model)
      objectStream.flush()
      byteStream.toByteArray
    } finally {
      objectStream.close()
      byteStream.close()
    }
  }

  def compile[ParamType <: FloatNN | ComplexNN: Default](
      model: TensorModule[ParamType],
      modelPath: String
  ): CompilationUnit =
    val archive = new OutputArchive
    model.save(archive)
    archive.save_to(modelPath)
    val compileUnit = torchNative.compile(modelPath)
    compileUnit

  def freeze(scriptModule: JitModule): JitModule = torchNative.freeze(scriptModule)

  def freeze(
      scriptModule: JitModule,
      preservedAttrs: Seq[String],
      optimizeNumerics: Boolean = true
  ): JitModule = {
    val strVector = new StringVector(preservedAttrs*)
    val opt = new StringVectorOptional(strVector)
    torchNative.freeze(scriptModule, opt, optimizeNumerics)
  }

  def getFusionGroupInlining(): Boolean = torchNative.getFusionGroupInlining()

  def getGraphExecutorOptimize(): Boolean = torchNative.getGraphExecutorOptimize()

  def setGraphExecutorOptimize(optimize: Boolean): Unit =
    torchNative.setGraphExecutorOptimize(optimize)

  def kNextDirection(): Int = torchNative.kNextDirection()

  def kPrevDirection(): Int = torchNative.kPrevDirection()

  def ObjLoaderFunc(ptr: StrongTypePtr, value: IValue): Obj = torchNative.ObjLoaderFunc(ptr, value)

  def load(modelFilePath: String): JitModule = torchNative.load(modelFilePath)

  def load(modelPath: String, device: String, weightsOnly: Boolean): JitModule = {
    val deviceOpt: DeviceOptional = new DeviceOptional(new Device(device))
    torchNative.load(modelPath, deviceOpt, weightsOnly)
  }

  def from_pretrain_model(modelFilePath: String): JitModule = torchNative.load(modelFilePath)

  def from_pretrain_model(modelPath: String, device: String, weightsOnly: Boolean): JitModule = {
    val deviceOpt: DeviceOptional = new DeviceOptional(new Device(device))
    torchNative.load(modelPath, deviceOpt, weightsOnly)
  }

  def from_pretrain_model(
      modelPath: String,
      device: String,
      extrafilesMap: ExtraFilesMap,
      weightsOnly: Boolean
  ): JitModule = {
    val deviceOpt: DeviceOptional = new DeviceOptional(new Device(device))
    torchNative.load(modelPath, deviceOpt, extrafilesMap, weightsOnly)
  }

  def load(
      modelPath: String,
      device: String,
      extrafilesMap: ExtraFilesMap,
      weightsOnly: Boolean
  ): JitModule = {
    val deviceOpt: DeviceOptional = new DeviceOptional(new Device(device))
    torchNative.load(modelPath, deviceOpt, extrafilesMap, weightsOnly)
  }

  def load(
      modelStream: ReadAdapterInterface,
      device: String,
      extrafilesMap: ExtraFilesMap,
      weightsOnly: Boolean
  ): JitModule = {
    val deviceOpt: DeviceOptional = new DeviceOptional(new Device(device))
    torchNative.load(modelStream, deviceOpt, extrafilesMap, weightsOnly)
  }

  def getFusionStrategy(): FusionStrategy = torchNative.getFusionStrategy()

  def setFusionStrategy(fusionStrategy: FusionStrategy): FusionStrategy =
    torchNative.setFusionStrategy(fusionStrategy)

  def isBlockListedSchema(funcSchema: FunctionSchema): Boolean =
    torchNative.isBlockListedSchema(funcSchema)

  def parseSchema(layerName: String, bool: Boolean): FunctionSchema =
    torchNative.parseSchema(layerName, bool)

  def parseSchema(layerName: String): FunctionSchema = torchNative.parseSchema(layerName)

  def optimize_for_inference(scriptModule: JitModule): JitModule =
    torchNative.optimize_for_inference(scriptModule)

  def unpackOutputs(vecSeq: ValueVector): ValueVector = torchNative.unpackOutputs(vecSeq)

  def findAllNodes(graph: Graph, symbol: Symbol, bool: Boolean) =
    torchNative.findAllNodes(graph, symbol, bool)

  def findAllNodes(block: Block, symbol: Symbol, bool: Boolean) =
    torchNative.findAllNodes(block, symbol, bool)

  def findAllNodes(blockSeq: Seq[Block], symbol: Symbol, bool: Boolean) = {
    val ref = new BlockArrayRef()
    blockSeq.foreach(block => ref.put(block))
    torchNative.findAllNodes(ref, symbol, bool)
  }

  def replaceBlockWithFallbackGraph(block: Block, vec: ValueVector): JitNode = {

    torchNative.replaceBlockWithFallbackGraph(block, vec)
  }

  def parse_and_initialize_jit_module(
      modelStream: ByteBuffer,
      size: Long,
      extraFilesMap: ExtraFilesMap
  ): JitModule = {
    // public static native JitModule parse_and_initialize_jit_module(@Cast({"char*"}) @SharedPtr ByteBuffer var0, @Cast({"size_t"}) long var1, @ByRef ExtraFilesMap var3, @ByVal(nullValue = "std::optional<at::Device>(std::nullopt)") DeviceOptional var4);
    torchNative.parse_and_initialize_jit_module(modelStream, size, extraFilesMap)

  }

  def load_jit_module_from_file(fileName: String, extraFilesMap: ExtraFilesMap): JitModule = {
    //   public static native JitModule load_jit_module_from_file(@StdString String var0, @ByRef ExtraFilesMap var1, @ByVal(nullValue = "std::optional<at::Device>(std::nullopt)") DeviceOptional var2);
    ///    public static native JitModule load_jit_module_from_file(@StdString String var0, @ByRef ExtraFilesMap var1);
    torchNative.load_jit_module_from_file(fileName, extraFilesMap)

  }

  def load_jit_module_from_stream(
      inputStream: Pointer,
      extraFilesMap: ExtraFilesMap,
      device: Option[String] = None
  ): JitModule = {
    //  public static native JitModule load_jit_module_from_stream(@Cast({"std::istream*"}) @ByRef Pointer var0, @ByRef ExtraFilesMap var1, @ByVal(nullValue = "std::optional<at::Device>(std::nullopt)") DeviceOptional var2);
    //    public static native JitModule load_jit_module_from_stream(@Cast({"std::istream*"}) @ByRef Pointer var0, @ByRef ExtraFilesMap var1);
    //    val streamPointer: Pointer = None
    //    device match {
    //      case None => torchNative.load_jit_module_from_stream(streamPointer ,extraFilesMap)
    //    }
    //    val deviceOpt:DeviceOptional = new DeviceOptional(new Device(device))
    torchNative.load_jit_module_from_stream(inputStream, extraFilesMap)
  }

  def LintGraph(graph: Graph): Unit = torchNative.LintGraph(graph)

  def insertGraph(
      graph: Graph,
      insertGraph: Graph,
      vec: ValueVector,
      map: ValueValueMap
  ): ValueVector = {
    torchNative.insertGraph(graph, insertGraph, vec)
  }

  def insertGraph(graph: Graph, insertGraph: Graph, vec: ValueVector): ValueVector = {
    torchNative.insertGraph(graph, insertGraph, vec)
  }

  def registerOperator(operator: Operator): Unit = torchNative.registerOperator(operator)

  def getNodesModuleHierarchy(jitNode: JitNode): String = {
    torchNative.getNodesModuleHierarchy(jitNode).toString
  }

  def getAllOperators(): OperatorVector = {
    val operatorVector = torchNative.getAllOperators()
    operatorVector
  }

  def getAllOperatorsFor(symbol: Symbol): OperatorVector = {
    val operatorVector = torchNative.getAllOperatorsFor(symbol)
    operatorVector
  }

  def getAllSortedOperatorsFor(symbol: Symbol): OperatorVector = {

    val operatorVector = torchNative.getAllSortedOperatorsFor(symbol)
    operatorVector
  }

  def findSimilarOperators(symbol: Symbol): SymbolVector = {
    torchNative.findSimilarOperators(symbol)
  }

  def matchSchema(
      funcSchema: FunctionSchema,
      sourceRange: SourceRange,
      graph: Graph,
      nameValueSeq: Seq[NamedValue],
      nameValueSeq2: Seq[NamedValue]
  ): MatchedSchema = {
    val ref = new NamedValueArrayRef()
    nameValueSeq.foreach(nv => ref.put(nv))
    val ref2 = new NamedValueArrayRef()
    nameValueSeq2.foreach(nv => ref2.put(nv))
    torchNative.matchSchema(funcSchema, sourceRange, graph, ref, ref2)
  }

  def matchSchemas(
      funcSchemaSeq: Seq[FunctionSchema],
      sourceRange: SourceRange,
      graph: Graph,
      nameValueSeq: Seq[NamedValue],
      nameValueSeq2: Seq[NamedValue]
  ): SizeTMatchedSchemaPair = {
    val fsVec = new FunctionSchemaVector()
    for (elem <- funcSchemaSeq) {
      fsVec.put(elem)
    }
    val ref = new NamedValueArrayRef()
    nameValueSeq.foreach(nv => ref.put(nv))
    val ref2 = new NamedValueArrayRef()
    nameValueSeq2.foreach(nv => ref2.put(nv))
    torchNative.matchSchemas(fsVec, sourceRange, graph, ref, ref2)
  }

  def emitBuiltinCall(
      sourceRange: SourceRange,
      graph: Graph,
      symbol: Symbol,
      nameValueSeq: Seq[NamedValue],
      nameValueSeq2: Seq[NamedValue]
  ): Value = {
    val ref = new NamedValueArrayRef()
    nameValueSeq.foreach(nv => ref.put(nv))
    val ref2 = new NamedValueArrayRef()
    nameValueSeq2.foreach(nv => ref2.put(nv))
    torchNative.emitBuiltinCall(sourceRange, graph, symbol, ref, ref2)
  }

  def optimize_for_inference(scriptModule: JitModule, parameterSeq: Seq[String]): JitModule = {
    val strVector = new StringVector(parameterSeq*)
    torchNative.optimize_for_inference(scriptModule, strVector)
  }

  def getWriteableTensorData(tensor: PytorchTensor, bool: Boolean): WriteableTensorData =
    torchNative.getWriteableTensorData(tensor, bool)

  def findOperatorFor(operatorName: String, overloadName: String): Operator = {
    val opname = new BytePointer(operatorName)
    val ovname = new BytePointer(overloadName)
    import org.bytedeco.pytorch.OperatorName
    val operatorNamez: OperatorName = new OperatorName(opname, ovname)
    val operator = torchNative.findOperatorFor(operatorNamez)
    operator
  }

  def peek(ivalueSeq: Seq[IValue], l1: Long, l2: Long): IValue = {
    torchNative.peek(new IValueVector(ivalueSeq*), l1, l2)
  }

  def peekSlice(ivalueSeq: Seq[IValue], l1: Long, l2: Long, l3: Long) = {
    torchNative.peekSlice(new IValueVector(ivalueSeq*), l1, l2, l3)
  }

  def pop(ivalueSeq: Seq[IValue]): IValue = {
    torchNative.pop(new IValueVector(ivalueSeq*))
  }

  def pop(ivalueSeq: Seq[IValue], l: Long): IValueVector = {
    torchNative.pop(new IValueVector(ivalueSeq*), l)
  }

  def drop(ivalueSeq: Seq[IValue], l1: Long): Unit = {
    torchNative.drop(new IValueVector(ivalueSeq*), l1)
  }

  def last(ivalueSeq: Seq[IValue], l1: Long) = {
    torchNative.last(new IValueVector(ivalueSeq*), l1)
  }

  def push_one(ivalueSeq: Seq[IValue], tensorOpt: TensorOptions): Unit = {
    torchNative.push_one(new IValueVector(ivalueSeq*), tensorOpt)
  }

  //  public static native void
  //  backward(
  //  TensorVector var0,
  //  TensorVector var1,
  //  BoolOptional var2,
  //  boolean var3,
  //  TensorVector var4);

  def tensorVectorToSeqTensor2[D1 <: DType](vec: TensorVector): Seq[Tensor[D1]] = {
    var it = vec.begin()
    val tensorSeq = new ListBuffer[Tensor[D1]]()
    while (!it.equals(vec.end())) {
      val tensor = fromNative(it.get()).asInstanceOf[Tensor[D1]]
      tensorSeq.append(tensor)
      it = it.increment()
    }
    tensorSeq.toSeq
  }

  object autograd {
    def backward[D <: DType](tensors: Seq[Tensor[D]]): Unit = {
      torchNative.backward(
        new TensorVector(tensors.map(_.native)*)
      ) // new TensorVector(gradSeq.map(_.native)*))
    }

    def backward[D <: DType](
        tensors: Seq[Tensor[D]],
        grad_tensors: Seq[Tensor[D]],
        retain_graph: BoolOptional,
        create_graph: Boolean = false,
        inputs: Seq[Tensor[D]]
    ): Unit = {
      val tensorVector = torchNative.backward(
        new TensorVector(tensors.map(_.native)*),
        new TensorVector(grad_tensors.map(_.native)*),
        retain_graph,
        create_graph,
        new TensorVector(inputs.map(_.native)*)
      )
    }

    //    TensorVector grad(
    //    TensorVector var0,  TensorVector var1,
    //     TensorVector var2,
    //    BoolOptional var3, boolean var4,boolean var5);
    //
    //
    //    public static native TensorVector grad(TensorVector var0,
    //    TensorVector var1);
    def grad[D <: DType](outputs: Seq[Tensor[D]], inputs: Seq[Tensor[D]]): Seq[Tensor[D]] = {
      val vec = torchNative.grad(
        new TensorVector(outputs.map(_.native)*),
        new TensorVector(inputs.map(_.native)*)
      )
      tensorVectorToSeqTensor2(vec)
    }

    def grad[D <: DType](
        outputs: Seq[Tensor[D]],
        inputs: Seq[Tensor[D]],
        grad_outputs: Seq[Tensor[D]],
        retain_graph: Option[BoolOptional] = None,
        create_graph: Boolean = false,
        allow_unused: Boolean = false
    ): Seq[Tensor[D]] = {
      val nativeRetainGraph =
        if (retain_graph.isDefined) new BoolOptional(retain_graph.get) else new BoolOptional()
      val tensorVector = torchNative.grad(
        new TensorVector(outputs.map(_.native)*),
        new TensorVector(inputs.map(_.native)*),
        new TensorVector(grad_outputs.map(_.native)*),
        nativeRetainGraph,
        create_graph,
        allow_unused
      )
      tensorVectorToSeqTensor2(tensorVector)
    }

    def enter_dual_level(): Long = {
      torchNative.enter_dual_level()
    }

    def exit_dual_level(level: Long): Unit = {
      torchNative.exit_dual_level(level)
    }

  }
}
//object ModelSerializer {
//  // 将 Module 序列化为 BytePointer
//  def serializeModel[ParamType <: FloatNN | ComplexNN: Default](model: TensorModule[ParamType],tmpSavePath: String = "net.pt"): BytePointer = {
//    val buffer = new ByteBuffer()
//    val bytePointer = new BytePointer(ByteBuffer.wrap(bytes))
//    model.save(buffer) // 导出到内存流
////    val bytePointer = new BytePointer(buffer.size())
//    buffer.read(bytePointer, buffer.size()) // 读取字节流
//
//    val archive = new OutputArchive
//    model.save(archive)
//    archive.save_to(tmpSavePath)
//    bytePointer
//  }
//
//  // 从 BytePointer 加载模型
//  def deserializeModel[ParamType <: FloatNN | ComplexNN: Default](bytePointer: BytePointer): TensorModule[ParamType] = {
//    val buffer = new MemoryInputStream(bytePointer)
//    val module = new TensorModule()
//    module.load(buffer) // 加载模型
//    module
//  }
//}
