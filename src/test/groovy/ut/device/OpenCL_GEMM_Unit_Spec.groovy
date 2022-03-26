package ut.device

import neureka.Tsr
import neureka.backend.api.ExecutionCall
import neureka.backend.standard.operations.linear.internal.opencl.GEMM
import neureka.devices.opencl.KernelCaller
import neureka.devices.opencl.OpenCLDevice
import spock.lang.Specification


class OpenCL_GEMM_Unit_Spec extends Specification {

    def 'The GEMM implementation for the OpenCLDevice has realistic behaviour'()
    {
        given :
            var a = Tsr.ofFloats().withShape(3, 4).all(7)
            var b = Tsr.ofFloats().withShape(4, 2).all(-5)
            var c = Tsr.ofFloats().withShape(3, 2).all(0)
            var call = Mock(ExecutionCall)
            var device = Mock(OpenCLDevice)
            var kernel = Mock(KernelCaller)

        when :
            new GEMM().run( call )

        then :
            (1.._) * call.input(Float, 0) >> c
            (1.._) * call.input(Float, 1) >> a
            (1.._) * call.input(Float, 2) >> b
            (1.._) * call.getDevice() >> device
            (1.._) * device.hasAdHocKernel("fast_CM_MM_3x4x2") >> false
            (1.._) * device.compileAdHocKernel("fast_CM_MM_3x4x2", _) >> device
            (1.._) * device.getAdHocKernel("fast_CM_MM_3x4x2") >> kernel
            (3.._) * kernel.pass(_) >> kernel
    }


}