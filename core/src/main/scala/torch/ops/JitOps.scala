package torch
package ops

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch as torchNative
import torch.nn.modules.TensorModule

import java.io.{ByteArrayOutputStream, ObjectOutputStream}
import java.nio.ByteBuffer

trait JitOps {

  def compile[ParamType <: FloatNN | ComplexNN : Default](model: TensorModule[ParamType]): CompilationUnit =
    val bytes = modelToBytes(model)
    val bytePointer = new BytePointer(ByteBuffer.wrap(bytes))
    torchNative.compile(bytePointer)

  def modelToBytes[ParamType <: FloatNN | ComplexNN : Default](model: TensorModule[ParamType]): Array[Byte] = {
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

  def compile[ParamType <: FloatNN | ComplexNN : Default](model: TensorModule[ParamType], modelPath: String): CompilationUnit =
    val archive = new OutputArchive
    model.save(archive)
    archive.save_to(modelPath)
    val compileUnit = torchNative.compile(modelPath)
    compileUnit
}


//  @Namespace("torch::jit")
  //  @SharedPtr("torch::jit::CompilationUnit")
  //  @ByVal
  //  public static native CompilationUnit compile(@StdString BytePointer var0);
//  @ByVal
//  public static native JitModule load_jit_module_from_file(
//  @StdString String var0
//  , @ByRef ExtraFilesMap var1
//  , @ByVal(nullValue = "std::optional<at::Device>(std::nullopt)") DeviceOptional var2
//  );
//
//  @Namespace("torch::jit")
//  @ByVal
//  public static native JitModule load_jit_module_from_file(
//  @StdString String var0
//  , @ByRef ExtraFilesMap var1
//  );
//
//  @Namespace("torch::jit")
//  @Namespace("torch")
  //  @ByVal
  //  @Cast({"std::vector<char>*"})
  //  public static native ByteVector pickle_save(@Const @ByRef IValue var0);
  //
  //  @Namespace("torch")
  //  @ByVal
  //  public static native IValue pickle_load(@Cast({"const std::vector<char>*"}) @ByRef ByteVector var0);

//  public static native JitModule load(
//  @Cast({
//    "std::istream*"
//  }) @ByRef Pointer var0
//  , @ByVal(nullValue = "std::optional<c10::Device>(std::nullopt)") DeviceOptional var1
//  , @Cast({
//    "bool"
//  }) boolean var2
//  );
//
//  @Namespace("torch::jit")
//  @ByVal
//  public static native JitModule load(
//  @Cast({
//    "std::istream*"
//  }) @ByRef Pointer var0
//  );
//
//  @Namespace("torch::jit")
//  @ByVal
//  public static native JitModule load(
//  @Cast({
//    "std::istream*"
//  }) @ByRef Pointer var0
//  , @ByVal DeviceOptional var1
//  , @ByRef ExtraFilesMap var2
//  , @Cast({
//    "bool"
//  }) boolean var3
//  );


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
