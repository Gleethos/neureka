package ut.device.internal


import neureka.Tensor
import neureka.backend.api.ExecutionCall
import neureka.backend.main.implementations.linear.CLDot
import neureka.backend.main.operations.linear.internal.opencl.CLReduce
import neureka.backend.main.operations.linear.internal.opencl.CLSum
import neureka.backend.main.operations.linear.internal.opencl.CLGEMM
import neureka.devices.opencl.KernelCaller
import neureka.devices.opencl.OpenCLDevice
import spock.lang.Specification
import spock.lang.Subject

@Subject([CLGEMM, CLReduce])
class OpenCL_Kernel_Unit_Spec extends Specification
{
    def 'The GEMM implementation for the OpenCLDevice has realistic behaviour'()
    {
        given :
            var a = Tensor.ofFloats().withShape(3, 4).all(7)
            var b = Tensor.ofFloats().withShape(4, 2).all(-5)
            var c = Tensor.ofFloats().withShape(3, 2).all(0)
            var call = Mock(ExecutionCall)
            var device = Mock(OpenCLDevice)
            var kernel = Mock(KernelCaller)

        when :
            new CLGEMM().run( call )

        then :
            (1.._) * call.input(Float, 0) >> c
            (1.._) * call.input(Float, 1) >> a
            (1.._) * call.input(Float, 2) >> b
            (1.._) * call.getDevice() >> device
            (1.._) * device.hasAdHocKernel("fast_CM_MM_3x4x2") >> false
            (0.._) * device.compileAdHocKernel("fast_CM_MM_3x4x2", _) >> device
            (0.._) * device.compileAndGetAdHocKernel("fast_CM_MM_3x4x2", _) >> kernel
            (0.._) * device.getAdHocKernel("fast_CM_MM_3x4x2") >> kernel
            (3.._) * kernel.pass(_) >> kernel
    }


    def 'The Reduce implementation for the OpenCLDevice has realistic behaviour'(CLReduce.Type type)
    {
        given :
            var a = Tensor.ofFloats().withShape(19, 7).andWhere({ i, _ -> (1+(7**i)%30)})
            var call = Mock(ExecutionCall)
            var device = Mock(OpenCLDevice)
            var kernel = Mock(KernelCaller)

        when :
            new CLReduce(type).run( call )

        then :
            _ * call.input(0) >> a
            (1.._) * call.input(Float, 0) >> a
            (1.._) * call.getDevice() >> device
            (1.._) * device.maxWorkGroupSize() >> 64
            (0.._) * device.hasAdHocKernel("fast_${type.name().toLowerCase()}_reduce_RTS64") >>> [false, true]
            (0.._) * device.findAdHocKernel("fast_${type.name().toLowerCase()}_reduce_RTS64") >> Optional.of(kernel)
            (0.._) * device.compileAdHocKernel("fast_${type.name().toLowerCase()}_reduce_RTS64", _) >> device
            (0.._) * device.findOrCompileAdHocKernel("fast_${type.name().toLowerCase()}_reduce_RTS64", _) >> kernel
            (0.._) * device.compileAndGetAdHocKernel("fast_${type.name().toLowerCase()}_reduce_RTS64", _) >> kernel
            (0.._) * device.getAdHocKernel("fast_${type.name().toLowerCase()}_reduce_RTS64") >> kernel
            (3.._) * kernel.pass(_) >> kernel
        and :
            (0.._) * device.hasAdHocKernel(CLReduce.INDICES_MAPPER_ID) >>> [false, true]
            (0.._) * device.compileAdHocKernel(CLReduce.INDICES_MAPPER_ID, _) >> device
            (0.._) * device.compileAndGetAdHocKernel(CLReduce.INDICES_MAPPER_ID, _) >> kernel
            (0.._) * device.findOrCompileAdHocKernel(CLReduce.INDICES_MAPPER_ID, _) >> kernel
            (0.._) * device.getAdHocKernel(CLReduce.INDICES_MAPPER_ID) >> kernel

        where :
            type << [CLReduce.Type.MIN, CLReduce.Type.MAX]
    }


    def 'The Sum implementation for the OpenCLDevice has realistic behaviour'()
    {
        given :
            var a = Tensor.ofFloats().withShape(19, 7).andWhere({ i, _ -> (1+(7**i)%30)})
            var call = Mock(ExecutionCall)
            var device = Mock(OpenCLDevice)
            var kernel = Mock(KernelCaller)

        when :
            new CLSum().run( call )

        then :
            _ * call.input(0) >> a
            (1.._) * call.input(Float, 0) >> a
            (1.._) * call.getDevice() >> device
            (1.._) * device.maxWorkGroupSize() >> 32
            (0.._) * device.hasAdHocKernel("fast_local_mem_based_sum") >>> [false, true]
            (0.._) * device.compileAdHocKernel("fast_local_mem_based_sum", _) >> device
            (0.._) * device.compileAndGetAdHocKernel("fast_local_mem_based_sum", _) >> kernel
            (0.._) * device.findAdHocKernel("fast_local_mem_based_sum") >> Optional.of(kernel)
            (0.._) * device.getAdHocKernel("fast_local_mem_based_sum") >> kernel
            (0.._) * device.findOrCompileAdHocKernel("fast_local_mem_based_sum", _) >> kernel
            (3.._) * kernel.pass(_) >> kernel
            (1.._) * kernel.passLocalFloats(_) >> kernel
    }


    def 'The Sum implementation for the OpenCLDevice has realistic behaviour for when the number of elements is a prime.'()
    {
        given :
            var a = Tensor.ofFloats().withShape(31).andWhere({ i, _ -> (1+(7**i)%30)})
            var call = Mock(ExecutionCall)
            var device = Mock(OpenCLDevice)
            var kernel = Mock(KernelCaller)

        when :
            new CLSum().run( call )

        then :
            _ * call.input(0) >> a
            (1.._) * call.input(Float, 0) >> a
            (1.._) * call.getDevice() >> device
            (1.._) * device.maxWorkGroupSize() >> 16
            (0.._) * device.hasAdHocKernel("fast_private_sum_reduction_RTS16") >>> [false, true]
            (0.._) * device.compileAdHocKernel("fast_private_sum_reduction_RTS16", _) >> device
            (0.._) * device.compileAndGetAdHocKernel("fast_private_sum_reduction_RTS16", _) >> kernel
            (0.._) * device.findAdHocKernel("fast_private_sum_reduction_RTS16") >> Optional.of(kernel)
            (0.._) * device.getAdHocKernel("fast_private_sum_reduction_RTS16") >> kernel
            (0.._) * device.findOrCompileAdHocKernel("fast_private_sum_reduction_RTS16", _) >> kernel
            (0.._) * device.hasAdHocKernel("fast_local_mem_based_sum") >>> [false, true]
            (0.._) * device.compileAdHocKernel("fast_local_mem_based_sum", _) >> device
            (0.._) * device.compileAndGetAdHocKernel("fast_local_mem_based_sum", _) >> kernel
            (0.._) * device.findAdHocKernel("fast_local_mem_based_sum") >> Optional.of(kernel)
            (0.._) * device.getAdHocKernel("fast_local_mem_based_sum") >> kernel
            (0.._) * device.findOrCompileAdHocKernel("fast_local_mem_based_sum", _) >> kernel
            (3.._) * kernel.pass(_) >> kernel
            (1.._) * kernel.passLocalFloats(_) >> kernel
    }

    def 'The CLDot implementation for the OpenCLDevice has realistic behaviour'()
    {
        given :
            var a = Tensor.ofFloats().withShape(19).andWhere({ i, _ -> (1+(7**i)%30)})
            var b = Tensor.ofFloats().withShape(19).andWhere({ i, _ -> (1+(7**i)%30)})
            var c = Tensor.ofFloats().withShape(1).all(0)
            var call = Mock(ExecutionCall)
            var device = Mock(OpenCLDevice)
            var kernel = Mock(KernelCaller)

        when :
            new CLDot().run( call )

        then :
            _ * call.input(0) >> c
            _ * call.input(1) >> a
            _ * call.input(2) >> b
            _ * call.input(Float.class, 0) >> c
            _ * call.input(Float.class, 1) >> a
            _ * call.input(Float.class, 2) >> b
            (1.._) * call.getDevice() >> device
            (1.._) * device.maxWorkGroupSize() >> 32
            (0.._) * device.hasAdHocKernel("multiply_arrays_for_dot_product") >>> [false, true]
            (0.._) * device.compileAdHocKernel("multiply_arrays_for_dot_product", _) >> device
            (0.._) * device.compileAndGetAdHocKernel("multiply_arrays_for_dot_product", _) >> kernel
            (0.._) * device.findAdHocKernel("multiply_arrays_for_dot_product") >> Optional.of(kernel)
            (0.._) * device.getAdHocKernel("multiply_arrays_for_dot_product") >> kernel
            (0.._) * device.findOrCompileAdHocKernel("multiply_arrays_for_dot_product", _) >> kernel
            (0.._) * device.hasAdHocKernel("fast_local_mem_based_sum") >>> [false, true]
            (0.._) * device.compileAdHocKernel("fast_local_mem_based_sum", _) >> device
            (0.._) * device.compileAndGetAdHocKernel("fast_local_mem_based_sum", _) >> kernel
            (0.._) * device.findAdHocKernel("fast_local_mem_based_sum") >> Optional.of(kernel)
            (0.._) * device.getAdHocKernel("fast_local_mem_based_sum") >> kernel
            (0.._) * device.findOrCompileAdHocKernel("fast_local_mem_based_sum", _) >> kernel
            (3.._) * kernel.pass(_) >> kernel
            (1.._) * kernel.passLocalFloats(_) >> kernel
    }

}