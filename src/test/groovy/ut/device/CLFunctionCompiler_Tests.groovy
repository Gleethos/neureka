package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.backend.standard.algorithms.GenericAlgorithm
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionBuilder
import neureka.devices.opencl.KernelCaller
import neureka.devices.opencl.OpenCLDevice
import neureka.devices.opencl.utility.CLFunctionCompiler
import spock.lang.Specification

class CLFunctionCompiler_Tests extends Specification {

    def 'The CLFunctionCompiler produces an operation which properly integrates the backend.'() {

        given :
            def mockDevice = Mock(OpenCLDevice)
        and :
            Tsr<Number> t1 = Tsr.of(1)
            Tsr<Number> t2 = Tsr.of(-2)
            Tsr<Number> t3 = Tsr.of(5)
        and :
            def funToBeOptimized = Function.create("i2 - (i0 / i1)")
        and :
            def compiler = new CLFunctionCompiler(
                                            mockDevice,
                                            funToBeOptimized,
                                            "test_fun"
                                        )

        when :
            Operation resultOperation = compiler.optimize()

        then :
            resultOperation != null

        when :
            Function fun = null
        and :
            def context = Neureka.get().context().clone().addOperation(resultOperation)
        and :
            context.runner()
                        .run {
                            fun = new FunctionBuilder(Neureka.get().context())
                                        .build(resultOperation, 3, true)
                        }

        then :
            fun != null
        and :
            fun.toString() == "test_fun(I[0], I[1], I[2])"

        when :
            def result = fun( t1, t2, t3 )

        then :
            def exception = thrown(IllegalStateException)
            exception.message == "No suitable implementation found for algorithm 'generic_algorithm_for_test_fun' and device type 'HostCPU'."

        when :
            t1.set(mockDevice)
            t2.set(mockDevice)
            t3.set(mockDevice)

        then :
            (2.._) * mockDevice.has(t1) >>> [false, true] // doesn't have it, then storing it, then has it!
            (2.._) * mockDevice.has(t2) >>> [false, true]
            (2.._) * mockDevice.has(t3) >>> [false, true]


        and :
            t1.isOutsourced()
            t2.isOutsourced()
            t3.isOutsourced()

        when :
            result = fun( t1, t2, t3 )


        then :
            1 * mockDevice.execute({ ExecutionCall<?> call ->
                call.device == mockDevice &&
                call.operation == resultOperation
            })

    }


    def 'The CLFunctionCompiler produces the expected "ad hoc" kernel.'() {

        given :
            def mockDevice = Mock(OpenCLDevice)
        and :
            def mockCaller = Mock(KernelCaller)
        and :
            def funToBeOptimized = Function.create("i2 - (i0 / i1)")
        and :
            def compiler = new CLFunctionCompiler(
                                    mockDevice,
                                    funToBeOptimized,
                                    "test_fun"
                            )

        when :
            Operation resultOperation = compiler.optimize()
        and :
            GenericAlgorithm algorithm = resultOperation.getAlgorithm(GenericAlgorithm.class)

        then :
            resultOperation != null
        and :
            algorithm != null
        and :
            algorithm.getImplementationFor(OpenCLDevice.class) != null

        when :
            algorithm
                .getImplementationFor(OpenCLDevice.class)
                .run(
                        ExecutionCall.builder()
                                        .tensors(Tsr.of(0), Tsr.of(1), Tsr.of(2), Tsr.of(3))
                                        .operation(resultOperation)
                                        .algorithm(algorithm)
                                        .device(mockDevice)
                                        .build() as ExecutionCall<OpenCLDevice>
                )

        then :
            1 * mockDevice.hasAdHocKernel(_) >>> [false, true]
        and :
            1 * mockDevice.getAdHocKernel(_) >> mockCaller
        and :
            4 * mockCaller.pass(_)
        and :
            1 * mockCaller.call(1)

    }


}
