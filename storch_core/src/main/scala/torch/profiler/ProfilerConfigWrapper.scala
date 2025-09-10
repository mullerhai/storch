package torch.profiler

import org.bytedeco.pytorch.{ProfilerConfig, ExperimentalConfig, IValue}

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.pytorch.global.torch._

/** Scala 包装器类，用于封装 Java 的 ProfilerConfig 实现 保持所有方法的一一映射，并处理类型适配
  */
class ProfilerConfigWrapper(private val underlying: ProfilerConfig) {

  // region 构造函数映射
  /** 主构造函数 - 完整参数版本
    */
  def this(
      state: ProfilerState,
      reportInputShapes: Boolean = false,
      profileMemory: Boolean = false,
      withStack: Boolean = false,
      withFlops: Boolean = false,
      withModules: Boolean = false,
      experimentalConfig: ExperimentalConfig = new ExperimentalConfig(),
      traceId: String = ""
  ) = this(
    new ProfilerConfig(
      state,
      reportInputShapes,
      profileMemory,
      withStack,
      withFlops,
      withModules,
      experimentalConfig,
      new BytePointer(traceId)
    )
  )

  /** 简化构造函数 - 仅接受 ProfilerState
    */
  def this(state: ProfilerState) = this(new ProfilerConfig(state))

  /** 简化构造函数 - 接受 ProfilerState 整数表示
    */
  def this(state: Int) = this(new ProfilerConfig(state))
  // endregion

  // region 方法映射
  def disabled(): Boolean = underlying.disabled()

  def global(): Boolean = underlying.global()

  def pushGlobalCallbacks(): Boolean = underlying.pushGlobalCallbacks()

  // Getter/Setter 方法映射
  def state(): ProfilerState = underlying.state()
  def state_=(setter: ProfilerState): Unit = underlying.state(setter)

  def experimentalConfig(): ExperimentalConfig = underlying.experimental_config()
  def experimentalConfig_=(setter: ExperimentalConfig): Unit =
    underlying.experimental_config(setter)

  def reportInputShapes(): Boolean = underlying.report_input_shapes()
  def reportInputShapes_=(setter: Boolean): Unit = underlying.report_input_shapes(setter)

  def profileMemory(): Boolean = underlying.profile_memory()
  def profileMemory_=(setter: Boolean): Unit = underlying.profile_memory(setter)

  def withStack(): Boolean = underlying.with_stack()
  def withStack_=(setter: Boolean): Unit = underlying.with_stack(setter)

  def withFlops(): Boolean = underlying.with_flops()
  def withFlops_=(setter: Boolean): Unit = underlying.with_flops(setter)

  def withModules(): Boolean = underlying.with_modules()
  def withModules_=(setter: Boolean): Unit = underlying.with_modules(setter)

  def traceId(): String = underlying.trace_id().getString
  def traceId_=(setter: String): Unit = underlying.trace_id(new BytePointer(setter))

  // 序列化方法
  def toIValue(): IValue = underlying.toIValue()
  // endregion

  /** 获取底层 Java 实例（用于需要原始对象的场景）
    */
//  def underlying(): ProfilerConfig = underlying
}

/** 伴生对象，提供静态方法映射
  */
object ProfilerConfigWrapper {

  /** 静态方法 fromIValue 的映射
    */
  def fromIValue(profilerConfigIValue: IValue): ProfilerConfigWrapper =
    new ProfilerConfigWrapper(ProfilerConfig.fromIValue(profilerConfigIValue))
}
