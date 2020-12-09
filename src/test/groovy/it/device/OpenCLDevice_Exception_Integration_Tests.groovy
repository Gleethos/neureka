package it.device

import neureka.Neureka
import neureka.devices.opencl.OpenCLPlatform
import spock.lang.Specification

class OpenCLDevice_Exception_Integration_Tests extends Specification
{
    def 'Ad hoc compilation produces expected exceptions.'()
    {
        given : 'This system supports OpenCL'
            if ( !Neureka.instance().canAccessOpenCL() ) return
            def device = OpenCLPlatform.PLATFORMS()[0].devices[0]

        expect :
            !device.hasAdHocKernel( 'right_dummy_kernel_name' )

        when :
            device.compileAdHocKernel( 'right_dummy_kernel_name', """
                    __kernel void wrong_dummy_kernel_name (
                            __global float* output,
                            __global float* input,
                            float value 
                        ) { 
                            unsigned int i = get_global_id( 0 );
                            output[i] = input[i] + value; 
                        }
                    """
            )

        then :
            def exception = thrown( IllegalArgumentException )
            exception.message == "Method 'clCreateKernel' failed! The name of the '__kernel' method declared inside \n" +
                    "the source String does not match the provided name needed for kernel creation."

        and :
            !device.hasAdHocKernel( 'right_dummy_kernel_name' )

    }

    def 'Ad hoc compilation produces expected exceptions when duplication is found.'()
    {
        given : 'This system supports OpenCL'
            if ( !Neureka.instance().canAccessOpenCL() ) return
            def device = OpenCLPlatform.PLATFORMS()[0].devices[0]
            def code = """
                        __kernel void right_dummy_kernel_name (
                                __global float* output,
                                __global float* input,
                                float value 
                            ) { 
                                unsigned int i = get_global_id( 0 );
                                output[i] = input[i] + value; 
                            }
                        """

        expect :
            !device.hasAdHocKernel( 'right_dummy_kernel_name' )

        when :
            device.compileAdHocKernel( 'right_dummy_kernel_name', code )

        then :
            device.hasAdHocKernel( 'right_dummy_kernel_name' )

        when :
            device.compileAdHocKernel( 'right_dummy_kernel_name', code )


        then :
            def exception = thrown( IllegalArgumentException )
            exception.message == "Cannot compile kernel source for name 'right_dummy_kernel_name' because the name is already taken.\n" +
                    "Use another name or find out why this kernel already exists.\n" +
                    "Besides the name, the source code of the existing kernel is also identical.\n"

        and :
            device.hasAdHocKernel( 'right_dummy_kernel_name' )

    }



}
