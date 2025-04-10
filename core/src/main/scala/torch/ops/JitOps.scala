package torch
package ops

import org.bytedeco.pytorch.global.torch as torchNative
import internal.NativeConverters.fromNative
import org.bytedeco.pytorch.TensorVector
import torch.nn.modules.TensorModule
import org.bytedeco.javacpp.BytePointer
trait JitOps {

  def compile[ParamType <: FloatNN | ComplexNN: Default](model: TensorModule[ParamType]): TensorModule[ParamType] =
    torchNative.compile(BytePointer(model.))

  def compile[ParamType <: FloatNN | ComplexNN: Default](modelPath: String): Unit =
    fromNative(torchNative.compile(modelPath))

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

}
