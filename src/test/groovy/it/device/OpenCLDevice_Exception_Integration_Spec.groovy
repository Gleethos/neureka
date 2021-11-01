package it.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.opencl.CLContext
import neureka.dtype.DataType
import org.slf4j.Logger
import spock.lang.IgnoreIf
import spock.lang.Specification

class OpenCLDevice_Exception_Integration_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2> OpenCLDevice Exception Integration Tests </h2>
            <p>
                Specified below are strict tests covering the behavior
                of the OpenCLDevice when encountering exceptional situation
                where insightful error messages are important.
            </p>
        """
        Neureka.get().reset()
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'An OpenCLDevice will throw an exception when trying to add a tensor whose "data parent" is not outsourced.'()
    {
        given: 'The first found OpenCLDevice instance.'
            Device device = Device.find('first')
        and : 'A tensor and a slice tensor of the prior.'
            Tsr t = Tsr.of([4, 3], 2)
            Tsr s = t[1..3, 1..2]

        expect : 'Both tensors share not only the same data but also the same data type.'
            t.data == s.data
            t.dataType == DataType.of( Double.class )
            s.dataType == DataType.of( Double.class )

        when : 'We try to add the slice to the device.'
            device.store(s)

        then : 'An exception is being thrown.'
            def exception = thrown(IllegalStateException)

        and : 'It explains what went wrong.'
            exception.message=="Data parent is not outsourced!"
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc compilation produces expected exceptions.'()
    {
        given :
            def device = Neureka.get().context().get(CLContext.class).platforms[0].devices[0]

        expect : 'Initially there is no ad hoc kernel with the following name.'
            !device.hasAdHocKernel( 'right_dummy_kernel_name' )

        when : 'We try to compile a new ad hoc kernel named "right_dummy_kernel_name" containing the wrong name in source...'
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

        then : 'An exception is being raised because the kernel name provided does not match the one in the source.'
            def exception = thrown( IllegalArgumentException )
            exception.message == "Method 'clCreateKernel' failed! The name of the '__kernel' method declared inside \n" +
                    "the source String does not match the provided name needed for kernel creation."

        and : 'Still the kernel does not exist because it failed to compile.'
            !device.hasAdHocKernel( 'right_dummy_kernel_name' )

    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc compilation produces expected exceptions when duplication is found.'()
    {
        given :
            def device = Neureka.get().context().get(CLContext.class).getPlatforms()[0].devices[0]
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

        expect : 'Initially there is no ad hoc kernel with the following name.'
            !device.hasAdHocKernel( 'right_dummy_kernel_name' )

        when : 'We try to compile a new ad hoc kernel named "right_dummy_kernel_name"...'
            device.compileAdHocKernel( 'right_dummy_kernel_name', code )

        then : 'The compilation succeeds and the device stores the new ad hoc kernel.'
            device.hasAdHocKernel( 'right_dummy_kernel_name' )

        when : 'We try to compile the same kernel name and source again...'
            device.compileAdHocKernel( 'right_dummy_kernel_name', code )


        then : 'This leads to the following exception:'
            def exception = thrown( IllegalArgumentException )
            exception.message == "Cannot compile kernel source for name 'right_dummy_kernel_name' because the name is already taken.\n" +
                    "Use another name or find out why this kernel already exists.\n" +
                    "Besides the name, the source code of the existing kernel is also identical.\n"

        and : 'Of course the original kernel is still present.'
            device.hasAdHocKernel( 'right_dummy_kernel_name' )

    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Trying to restore a tensor which is not on a device raises exception.'()
    {
        given :
            def device = Neureka.get().context().get(CLContext.class).getPlatforms()[0].devices[0]
        and : 'We create a new mock logger for the OpenCL device.'
            def oldLogger = device._log
            device._log = Mock( Logger )

        when : 'We pass a new tensor to the restore method of the device, even though the tensor is not stored on it...'
            device.restore( Tsr.newInstance() )

        then : 'The previous attempt raises an illegal argument exception with an explanatory message.'
            def exception = thrown( IllegalArgumentException )
            exception.message == "The passed tensor cannot be restored from " +
                    "this OpenCL device because the tensor is not stored on the device.\n"

        and : 'This message is also being logged by the internal device logger.'
            1 * device._log.error(
                    "The passed tensor cannot be restored from " +
                    "this OpenCL device because the tensor is not stored on the device.\n"
            )

        cleanup : 'Afterwards we restore the original logger!'
            device._log = oldLogger
    }


}
