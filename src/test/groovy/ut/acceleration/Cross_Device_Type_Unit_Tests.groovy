package ut.acceleration

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.acceleration.host.HostCPU
import neureka.acceleration.opencl.OpenCLDevice
import neureka.calculus.backend.ExecutionCall
import neureka.calculus.backend.implementations.OperationTypeImplementation
import spock.lang.Specification


class Cross_Device_Type_Unit_Tests extends Specification
{

    def 'Querying for Device implementations works as expected.'(
            String query, Class type
    ) {
        given : 'This system supports OpenCL.'
            if ( !Neureka.instance().canAccessOpenCL() ) return

        and : 'Neureka is being reset.'
            Neureka.instance().reset()

        when : 'The query is being passed to the "find" method...'
            def device = Device.find(query)

        then : 'The resulting Device variable has the expected type.'
            device.class == type

        where :
            query                       || type
            "cPu"                       || HostCPU.class
            "jVm"                       || HostCPU.class
            "natiVe"                    || HostCPU.class
            "Threaded"                  || HostCPU.class
            "openCl"                    || OpenCLDevice.class
            "nvidia or amd or intel"    || OpenCLDevice.class // This assumes that there is an amd/intel/nvidia gpu!
            "first"                     || OpenCLDevice.class
    }


    /**
     * The data of a tensor located on an Device should
     * be update when passing a float or double array!
     */
    def 'Passing a numeric array to a tensor should modify its content!'(
            Device device, Object data1, Object data2, String expected
    ) {
        given : 'Neureka is being reset.'
            Neureka.instance().reset()
        and : 'A 2D tensor is being instantiated..'
            Tsr t = new Tsr(new int[]{3, 2}, new double[]{2, 4, -5, 8, 3, -2}).add(device)

        when : 'A numeric array is passed to said tensor...'
            if( data1 instanceof float[] ) t.setValue32(data1)
            else t.setValue64(data1 as double[])
            if( data2 instanceof float[] ) t.setValue32(data2)
            else t.setValue64(data2 as double[])

        then : 'The tensor (as String) contains the expected String.'
            t.toString().contains(expected)

        where : 'The following data is being used :'
            device                | data1                      | data2                      || expected
            Device.find("cpu")    | new float[0]               | new float[0]               || "(3x2):[2.0, 4.0, -5.0, 8.0, 3.0, -2.0]"
            Device.find("cpu")    | new float[]{2, 3, 4, 5, 6} | new float[]{1, 1, 1, 1, 1} || "(3x2):[1.0, 1.0, 1.0, 1.0, 1.0, -2.0]"
            Device.find("cpu")    | new float[]{3, 5, 6}       | new float[]{4, 2, 3}       || "(3x2):[4.0, 2.0, 3.0, 8.0, 3.0, -2.0]"
            Device.find("cpu")    | new double[]{9, 4, 7, -12} | new double[]{-5, -2, 1}    || "(3x2):[-5.0, -2.0, 1.0, -12.0, 3.0, -2.0]"
            Device.find("cpu")    | new float[]{22, 24, 35, 80}| new double[]{-1, -1, -1}   || "(3x2):[-1.0, -1.0, -1.0, 80.0, 3.0, -2.0]"

            Device.find("openCL") | new float[0]               | new float[0]               || "(3x2):[2.0, 4.0, -5.0, 8.0, 3.0, -2.0]"
            Device.find("openCL") | new float[]{2, 3, 4, 5, 6} | new float[]{1, 1, 1, 1, 1} || "(3x2):[1.0, 1.0, 1.0, 1.0, 1.0, -2.0]"
            Device.find("openCL") | new float[]{3, 5, 6}       | new float[]{4, 2, 3}       || "(3x2):[4.0, 2.0, 3.0, 8.0, 3.0, -2.0]"
            Device.find("openCL") | new double[]{9, 4, 7, -12} | new double[]{-5, -2, 1}    || "(3x2):[-5.0, -2.0, 1.0, -12.0, 3.0, -2.0]"
            Device.find("openCL") | new float[]{22, 24, 35, 80}| new double[]{-1, -1, -1}   || "(3x2):[-1.0, -1.0, -1.0, 80.0, 3.0, -2.0]"
    }

    /**
     *  Every argument within an ExecutionCall instance has a purpose. Null is not permissible.
     */
    def 'Execution calls containing null arguments will cause an exception to be thrown in device instances.'(
        Device device
    ) {

        given : 'The given device is available and Neureka is being reset.'
            if ( device == null ) return
            Neureka.instance().reset()
        and : 'A mocked ExecutionCall with mocked operation implementation and a mocked drain instantiator lambda...'
            def call = Mock(ExecutionCall)
            def implementation = Mock(OperationTypeImplementation)

        when : 'The call is being passed to the device for execution...'
            device.execute(call)

        then : '...the implementation is being accessed in order to access the mocked lambda...'
            1 * call.getImplementation() >> implementation
            1 * implementation.instantiateNewTensorsForExecutionIn(call) >> call
        and : 'The tensor array is being accessed to check for null. (For exception throwing)'
            1 * call.getTensors() >> new Tsr[]{ Mock(Tsr), null }
        and : 'The expected exception is being thrown alongside a descriptive message.'
            def exception = thrown(IllegalArgumentException)
            exception.message == "Device arguments may not be null!\n" +
                    "One or more tensor arguments within the given ExecutionCall instance is null."

        where : 'The following Device instances are being tested :'
            device << [
                    HostCPU.instance(),
                    Device.find("openCL")
            ]

    }


}