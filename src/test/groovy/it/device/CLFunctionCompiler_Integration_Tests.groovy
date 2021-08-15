package it.device

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.devices.opencl.CLContext
import spock.lang.Specification

class CLFunctionCompiler_Integration_Tests extends Specification {

    def setupSpec()
    {
        reportHeader """
            <h2> OpenCLDevice Function Optimization Integration Tests </h2>
            <p>
                Specified below are strict tests for covering the ability of 
                OpenCL devices to be able produce optimized functions given
                a normal function instance created from a String...
            </p>
        """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'The OpenCLDevice produces a working optimized Function (internally using the CLFunctionCompiler).'() {

        given : 'This system supports OpenCL'
            if ( !Neureka.get().canAccessOpenCL() ) return
        and : 'We get the first available OpenCLDevice we can find in the CLContext!'
            def device = Neureka.get().context().get(CLContext.class).platforms[0].devices[0]
        and : 'Three scalar test tensors which will be used as inputs to the optimized function.'
            Tsr<Double> t1 = Tsr.of(-2).set(device)
            Tsr<Double> t2 = Tsr.of(5).set(device)
            Tsr<Double> t3 = Tsr.of(2).set(device)

        and : 'A test function which will be the optimization target for this test.'
            def funToBeOptimized = Function.of("i2 + (i0 / i1)") // 2 + (-2 / 5)

        when : 'We instruct the device to produce an optimized Function based on the provided test function...'
            Function optimized = device.optimizedFunctionOf(funToBeOptimized, "my_test_fun")

        then : 'Initially we expect that the device does not contain the "ad hoc" kernel with the following signature...'
            !device.hasAdHocKernel("my_test_fun_F32\$1_F32\$1_F32\$1_F32\$1")

        when :
            Tsr result = optimized( t1, t2, t3 )

        then :
            result.toString() == "(1):[1.6E0]"

        and :
            device.hasAdHocKernel("my_test_fun_F32\$1_F32\$1_F32\$1_F32\$1")

    }


    /* // WIP
    def 'The OpenCLDevice produces an optimized Function for slices.'() {

        given : 'This system supports OpenCL'
        if ( !Neureka.get().canAccessOpenCL() ) return
        and : 'We get the first available OpenCLDevice we can find in the CLContext!'
        def device = Neureka.get().context().get(CLContext.class).platforms[0].devices[0]
        and : 'Three scalar test tensors which will be used as inputs to the optimized function.'
        Tsr<Double> t1 = Tsr.of([[1, 3, 2],[4, -2, 5]])[0..1, 1..2]
        t1.set(device)
        Tsr<Double> t2 = Tsr.of(5).set(device)
        Tsr<Double> t3 = Tsr.of(2).set(device)

        and : 'A test function which will be the optimization target for this test.'
        def funToBeOptimized = Function.of("i2 + (i0 / i1)") // 2 + (-2 / 5)

        when : 'We instruct the device to produce an optimized Function based on the provided test function...'
        Function optimized = device.optimizedFunctionOf(funToBeOptimized, "my_test_fun")

        then : 'Initially we expect that the device does not contain the "ad hoc" kernel with the following signature...'
        !device.hasAdHocKernel("my_test_fun_F32\$1_F32\$1_F32\$1_F32\$1")

        when :
        Tsr result = optimized( t1, t2, t3 )

        then :
        result.toString() == "(1):[1.6E0]"

        and :
        device.hasAdHocKernel("my_test_fun_F32\$1_F32\$1_F32\$1_F32\$1")

    }

     */

}
