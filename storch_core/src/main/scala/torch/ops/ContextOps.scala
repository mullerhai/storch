package torch
package ops

import org.bytedeco.pytorch.global.{torch as torchNative}
import org.bytedeco.pytorch.{
  DoubleOptional,
  LongOptional,
  PackedSequence,
  Scalar,
  ScalarOptional,
  TensorOptional,
  TensorOptionalList,
  TensorVector,
  Context,
  LongVector
}
import torch.internal.NativeConverters.{fromNative, toScalar}

import scala.collection.mutable.ListBuffer

private[torch] trait ContextOps {

  given context: Context = new Context()
  def setQEngine(engine: torchNative.QEngine)(using context: Context) = context.setQEngine(engine)
  def setSDPPriorityOrder(order: Array[Long])(using context: Context): Unit =
    context.setSDPPriorityOrder(new LongVector(order*))
  def setBenchmarkLimitCuDNN(limit: Int)(using context: Context) =
    context.setBenchmarkLimitCuDNN(limit)

  def hasOpenMP = Context.hasOpenMP()
  def hasMKL = Context.hasMKL()
  def hasKleidiAI = Context.hasKleidiAI()
  def hasLAPACK = Context.hasLAPACK()
  def hasMKLDNN = Context.hasMKLDNN()
  def hasMAGMA = Context.hasMAGMA()

  def hasCUDA = Context.hasCUDA()

  def hasMTIA = Context.hasMTIA()

  def hasCUDART = Context.hasCUDART()

  def hasCuDNN = Context.hasCuDNN()

  def hasCuSOLVER = Context.hasCuSOLVER()

  def hasCuBLASLt = Context.hasCuBLASLt()

  def hasROCM = Context.hasROCM()

  def hasHIP = Context.hasHIP()

  def hasMPS = Context.hasMPS()

  def hasIPU = Context.hasIPU()

  def hasXLA = Context.hasXLA()

  def hasXPU = Context.hasXPU()

  def hasLazy = Context.hasLazy()

  def hasMAIA = Context.hasMAIA()

  def hasHPU = Context.hasHPU()

  def versionCuDNN: Long = Context.versionCuDNN()

  def versionCUDART: Long = Context.versionCUDART()

  def getNVRTC = Context.getNVRTC()

  def lazyInitCUDA(using context: Context) = context.lazyInitCUDA()

  def lazyInitHIP(using context: Context) = context.lazyInitHIP()

  def lazyInitXPU(using context: Context) = context.lazyInitXPU()

  def lazyInitMTIA(using context: Context) = context.lazyInitMTIA()

  def lazyInitPrivateUse1(using context: Context) = context.lazyInitPrivateUse1()

  def unsetDefaultMobileCPUAllocator(using context: Context) =
    context.unsetDefaultMobileCPUAllocator()

  def setDefaultMobileCPUAllocator(using context: Context) = context.setDefaultMobileCPUAllocator()

  def qEngine(using context: Context) = context.qEngine()

  def supportedQEngines = Context.supportedQEngines()

  def alertCuBLASConfigNotDeterministic(using context: Context) =
    context.alertCuBLASConfigNotDeterministic()

  def setFloat32MatmulPrecision(precision: String)(using context: Context) =
    context.setFloat32MatmulPrecision(precision)

  def alertNotDeterministic(caller: String) = Context.alertNotDeterministic(caller)

  def deterministicFillUninitializedMemory(using context: Context) =
    context.deterministicFillUninitializedMemory()

  def deterministicAlgorithms(using context: Context) = context.deterministicAlgorithms()

  def deterministicAlgorithmsWarnOnly(using context: Context) =
    context.deterministicAlgorithmsWarnOnly()

  def float32MatmulPrecision(using context: Context) = context.float32MatmulPrecision()

  def getROCmFAPreferredBackend(using context: Context) = context.getROCmFAPreferredBackend()

  def blasPreferredBackend(using context: Context) = context.blasPreferredBackend()

  def userEnabledNNPACK(using context: Context) = context.userEnabledNNPACK()

  def sDPPriorityOrder(using context: Context) = context.sDPPriorityOrder()

//  def userEnabledNNPACK(using context: Context): Boolean = context.userEnabledNNPACK()

  def deterministicCuDNN(using context: Context) = context.deterministicCuDNN()

  def deterministicMkldnn(using context: Context): Boolean = context.deterministicMkldnn()

  def benchmarkLimitCuDNN(using context: Context) = context.benchmarkLimitCuDNN()

  def benchmarkCuDNN(using context: Context) = context.benchmarkCuDNN()

  def setBenchmarkCuDNN(enabled: Boolean)(using context: Context) =
    context.setBenchmarkCuDNN(enabled)

  def setDeterministicCuDNN(enabled: Boolean)(using context: Context) =
    context.setDeterministicCuDNN(enabled)

  def setDeterministicMkldnn(enabled: Boolean)(using context: Context) =
    context.setDeterministicMkldnn(enabled)

  def setUserEnabledNNPACK(enabled: Boolean)(using context: Context) =
    context.setUserEnabledNNPACK(enabled)

  def setUserEnabledCuDNN(enabled: Boolean)(using context: Context) =
    context.setUserEnabledCuDNN(enabled)

  def setUserEnabledMkldnn(enabled: Boolean)(using context: Context) =
    context.setUserEnabledMkldnn(enabled)

  def setSDPUseFlash(enabled: Boolean)(using context: Context) = context.setSDPUseFlash(enabled)

  def setSDPUseMemEfficient(enabled: Boolean)(using context: Context) =
    context.setSDPUseMemEfficient(enabled)

  def setSDPUseMath(enabled: Boolean)(using context: Context) = context.setSDPUseMath(enabled)

  def setSDPUseCuDNN(enabled: Boolean)(using context: Context) = context.setSDPUseCuDNN(enabled)

  def userEnabledCuDNN(using context: Context) = context.userEnabledCuDNN()

  def userEnabledMkldnn(using context: Context) = context.userEnabledMkldnn()

  def userEnabledFlashSDP(using context: Context) = context.userEnabledFlashSDP()

  def userEnabledMemEfficientSDP(using context: Context) = context.userEnabledMemEfficientSDP()

  def userEnabledMathSDP(using context: Context) = context.userEnabledMathSDP()

  def userEnabledCuDNNSDP(using context: Context) = context.userEnabledCuDNNSDP()

  def setAllowFP16BF16ReductionMathSDP(enabled: Boolean)(using context: Context) =
    context.setAllowFP16BF16ReductionMathSDP(enabled)

  def setSDPUseOverrideable(enabled: Boolean)(using context: Context) =
    context.setSDPUseOverrideable(enabled)

  def setDeterministicFillUninitializedMemory(enabled: Boolean)(using context: Context) =
    context.setDeterministicFillUninitializedMemory(enabled)

  def setDeterministicAlgorithms(enabled: Boolean, en: Boolean)(using context: Context) =
    context.setDeterministicAlgorithms(enabled, en)

  def setAllowTF32CuDNN(enabled: Boolean)(using context: Context) =
    context.setAllowTF32CuDNN(enabled)

  def setAllowTF32OneDNN(enabled: Boolean)(using context: Context) =
    context.setAllowTF32OneDNN(enabled)

  def setAllowTF32CuBLAS(enabled: Boolean)(using context: Context) =
    context.setAllowTF32CuBLAS(enabled)

  def setAllowFP16ReductionCuBLAS(enabled: Boolean)(using context: Context) =
    context.setAllowFP16ReductionCuBLAS(enabled)

  def setAllowBF16ReductionCuBLAS(enabled: Boolean)(using context: Context) =
    context.setAllowBF16ReductionCuBLAS(enabled)

  def setAllowFP16AccumulationCuBLAS(enabled: Boolean)(using context: Context) =
    context.setAllowFP16AccumulationCuBLAS(enabled)

  def setCheckSparseTensorInvariants(enabled: Boolean)(using context: Context) =
    context.setCheckSparseTensorInvariants(enabled)

  def setReleaseWeightsWhenPrepacking(enabled: Boolean)(using context: Context) =
    context.setReleaseWeightsWhenPrepacking(enabled)

  def setAllowFP16ReductionCPU(enabled: Boolean)(using context: Context) =
    context.setAllowFP16ReductionCPU(enabled)

  def allowFP16BF16ReductionMathSDP(using context: Context) =
    context.allowFP16BF16ReductionMathSDP()

  def userEnabledOverrideableSDP(using context: Context) = context.userEnabledOverrideableSDP()

  def allowTF32CuDNN(using context: Context) = context.allowTF32CuDNN()

  def allowTF32OneDNN(using context: Context) = context.allowTF32OneDNN()

  def allowTF32CuBLAS(using context: Context) = context.allowTF32CuBLAS()

  def allowFP16ReductionCuBLAS(using context: Context) = context.allowFP16ReductionCuBLAS()

  def allowBF16ReductionCuBLAS(using context: Context) = context.allowBF16ReductionCuBLAS()

  def isXNNPACKAvailable = Context.isXNNPACKAvailable()

  def checkSparseTensorInvariants(using context: Context) = context.checkSparseTensorInvariants()

  def releaseWeightsWhenPrepacking(using context: Context) = context.releaseWeightsWhenPrepacking()

  def areVmapFallbackWarningsEnabled(using context: Context) =
    context.areVmapFallbackWarningsEnabled()

  def isDefaultMobileCPUAllocatorSet(using context: Context) =
    context.isDefaultMobileCPUAllocatorSet()

  def allowFP16ReductionCPU(using context: Context) = context.allowFP16ReductionCPU()

}
