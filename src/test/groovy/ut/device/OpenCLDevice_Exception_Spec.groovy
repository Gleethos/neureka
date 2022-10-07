package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.backend.ocl.CLBackend
import neureka.dtype.DataType
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Shared
import spock.lang.Specification
import spock.lang.Title

@Title("OpenCLDevice Exception Handling")
@Narrative('''

    The OpenCLDevice class, one of many implementations of the Device interface, 
    represents physical OpenCL devices.
    This specification defines how instances of this class deal with exceptional information.

''')
class OpenCLDevice_Exception_Spec extends Specification
{

    @Shared def oldStream

    def setupSpec()
    {
        reportHeader """
            <p>
                It is important that an OpenCLDevice gives insightful error messages
                when encountering exceptional situations.
            </p>
        """
        Neureka.get().reset()
    }

    def setup() {
        oldStream = System.err
        System.err = Mock(PrintStream)
    }

    def cleanup() {
        System.err = oldStream
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'An OpenCLDevice will throw an exception when trying to add a tensor whose "data parent" is not outsourced.'()
    {
        given: 'The first found OpenCLDevice instance.'
            Device device = Device.get('first')
        and : 'A tensor and a slice tensor of the prior.'
            Tsr t = Tsr.of([4, 3], 2d)
            Tsr s = t[1..3, 1..2]

        expect : 'Both tensors share not only the same data but also the same data type.'
            t.mut.data.ref == s.mut.data.ref
            t.dataType == DataType.of( Double.class )
            s.dataType == DataType.of( Double.class )

        when : 'We try to add the slice to the device.'
            device.store(s)

        then : 'An exception is being thrown.'
            def exception = thrown(IllegalStateException)

        and : 'It explains what went wrong.'
            exception.message == "Data parent is not outsourced!"
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc compilation produces expected exceptions.'()
    {
        given :
            def device = Neureka.get().backend().get(CLBackend.class).platforms[0].devices[0]

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

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'Ad hoc compilation produces expected exceptions when duplication is found.'()
    {
        given :
            def device = Neureka.get().backend().get(CLBackend.class).getPlatforms()[0].devices[0]
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


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'Trying to restore a tensor which is not on a device raises exception.'()
    {
        given :
            def device = Neureka.get().backend().get(CLBackend.class).getPlatforms()[0].devices[0]

        when : 'We pass a new tensor to the restore method of the device, even though the tensor is not stored on it...'
            device.restore( Tsr.newInstance() )

        then : 'The previous attempt raises an illegal argument exception with an explanatory message.'
            def exception = thrown( IllegalArgumentException )
            exception.message == "The passed tensor cannot be restored from " +
                    "this OpenCL device because the tensor is not stored on the device.\n"

        and : 'This message is also being logged by the internal device logger.'
            1 * System.err.println( "[Test worker] ERROR neureka.devices.opencl.OpenCLDevice - The passed tensor cannot be restored from " +
                    "this OpenCL device because the tensor is not stored on the device.\n"
            )

    }


}
