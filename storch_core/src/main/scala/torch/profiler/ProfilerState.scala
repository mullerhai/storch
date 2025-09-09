package torch
package profiler

//package torch.profiler.impl

enum ProfilerState(val value: Int):
  case Disabled extends ProfilerState(0)
  case CPU extends ProfilerState(1)
  case CUDA extends ProfilerState(2)
  case NVTX extends ProfilerState(3)
  case ITT extends ProfilerState(4)
  case PRIVATEUSE1 extends ProfilerState(5)
  case KINETO extends ProfilerState(6)
  case KINETO_GPU_FALLBACK extends ProfilerState(7)
  case KINETO_PRIVATEUSE1_FALLBACK extends ProfilerState(8)
  case KINETO_ONDEMAND extends ProfilerState(9)
  case NUM_PROFILER_STATES extends ProfilerState(10)

  /**
   * 确保返回枚举的单例实例，逻辑与Java版`intern()`一致
   */
  def intern(): ProfilerState =
    ProfilerState.values.find(_.value == this.value).getOrElse(this)

  /**
   * 重写toString，返回interned实例的名称
   */
//  override def toString(): String = intern().name