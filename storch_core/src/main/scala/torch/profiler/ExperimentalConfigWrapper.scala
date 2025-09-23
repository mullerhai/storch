package torch.profiler

import org.bytedeco.pytorch.{ExperimentalConfig}

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.StringVector

/** Scala 包装器类，封装 Java 的 ExperimentalConfig 实现 映射所有构造函数、属性方法及操作符方法
  */
class ExperimentalConfigWrapper private (private val underlying: ExperimentalConfig) {

  // region 构造函数映射
  /** 完整参数构造函数（映射 Java 带默认值的构造函数）
    */
  def this(
      profilerMetrics: StringVector = new StringVector(),
      profilerMeasurePerKernel: Boolean = false,
      verbose: Boolean = false,
      performanceEvents: StringVector = new StringVector(),
      enableCudaSyncEvents: Boolean = false,
      adjustProfilerStep: Boolean = false,
      disableExternalCorrelation: Boolean = false,
      profileAllThreads: Boolean = false,
      captureOverloadNames: Boolean = false,
      adjustTimestamps: Boolean = false
  ) = this(
    new ExperimentalConfig(
      profilerMetrics,
      profilerMeasurePerKernel,
      verbose,
      performanceEvents,
      enableCudaSyncEvents,
      adjustProfilerStep,
      disableExternalCorrelation,
      profileAllThreads,
      captureOverloadNames,
      adjustTimestamps
    )
  )

  /** 默认构造函数（无参数）
    */
  def this() = this(new ExperimentalConfig())

  /** Pointer 构造函数（用于指针转换）
    */
  def this(p: Pointer) = this(new ExperimentalConfig(p))

  /** 数组分配构造函数（指定大小）
    */
  def this(size: Long) = this(new ExperimentalConfig(size))
  // endregion

  // region 核心方法映射
  /** 映射 Java 的 operator bool() 方法
    */
  def asBoolean(): Boolean = underlying.asBoolean()

  /** 映射 position 方法（数组元素定位）
    */
  def position(position: Long): ExperimentalConfigWrapper = {
    underlying.position(position)
    this
  }

  /** 映射 getPointer 方法（获取指定索引的数组元素）
    */
  def getPointer(i: Long): ExperimentalConfigWrapper =
    new ExperimentalConfigWrapper(underlying.getPointer(i))
  // endregion

  // region 属性方法映射（getter/setter）
  // StringVector 属性
  def profilerMetrics(): StringVector = underlying.profiler_metrics()
  def profilerMetrics_=(setter: StringVector): Unit = underlying.profiler_metrics(setter)

  def performanceEvents(): StringVector = underlying.performance_events()
  def performanceEvents_=(setter: StringVector): Unit = underlying.performance_events(setter)

  // 布尔属性
  def profilerMeasurePerKernel(): Boolean = underlying.profiler_measure_per_kernel()
  def profilerMeasurePerKernel_=(setter: Boolean): Unit =
    underlying.profiler_measure_per_kernel(setter)

  def verbose(): Boolean = underlying.verbose()
  def verbose_=(setter: Boolean): Unit = underlying.verbose(setter)

  def enableCudaSyncEvents(): Boolean = underlying.enable_cuda_sync_events()
  def enableCudaSyncEvents_=(setter: Boolean): Unit = underlying.enable_cuda_sync_events(setter)

  def adjustProfilerStep(): Boolean = underlying.adjust_profiler_step()
  def adjustProfilerStep_=(setter: Boolean): Unit = underlying.adjust_profiler_step(setter)

  def disableExternalCorrelation(): Boolean = underlying.disable_external_correlation()
  def disableExternalCorrelation_=(setter: Boolean): Unit =
    underlying.disable_external_correlation(setter)

  def profileAllThreads(): Boolean = underlying.profile_all_threads()
  def profileAllThreads_=(setter: Boolean): Unit = underlying.profile_all_threads(setter)

  def captureOverloadNames(): Boolean = underlying.capture_overload_names()
  def captureOverloadNames_=(setter: Boolean): Unit = underlying.capture_overload_names(setter)

  def adjustTimestamps(): Boolean = underlying.adjust_timestamps()
  def adjustTimestamps_=(setter: Boolean): Unit = underlying.adjust_timestamps(setter)
  // endregion

  /** 获取底层 Java 实例
    */
//  def underlying(): ExperimentalConfig = underlying
}
