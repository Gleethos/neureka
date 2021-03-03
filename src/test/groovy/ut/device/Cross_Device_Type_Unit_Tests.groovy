package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.HostCPU
import neureka.devices.opencl.OpenCLDevice
import neureka.backend.api.ExecutionCall
import neureka.backend.api.algorithms.Algorithm
import neureka.devices.file.FileDevice
import spock.lang.Specification


class Cross_Device_Type_Unit_Tests extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2> Cross Device-Type Unit Tests </h2>
            <p>
                Specified below are strict tests for the factory methods in the
                Device interface as well as its various implementations 
                which should adhere to a certain set of common behaviours.
            </p>
        """
    }

    def setup() {
        Neureka.instance().reset()
        // Configure printing of tensors to be more compact:
        Neureka.instance().settings().view().asString = "dgc"
    }

    def 'Querying for Device implementations works as expected.'(
            String query, Class type
    ) {
        given : 'This system supports OpenCL.'
            if ( !Neureka.instance().canAccessOpenCL() ) return

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
        given : 'A 2D tensor is being instantiated..'
            Tsr t = new Tsr<>(new int[]{3, 2}, new double[]{2, 4, -5, 8, 3, -2}).set(device)

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

    def 'Tensor data can be fetched from device if the tensor is stored on it...'(
            Device device, int[] shape, Object data, List<Float> expected
    ) {
        given : 'Because in some environments OpenCL might not be available, the test will be stopped!'
            if ( device == null ) return

        when : 'A 2D tensor is being instantiated by passing the given shape and data...'
            Tsr t = new Tsr<>(shape, data).set(device)

        then : 'The tensor values (as List) are as expected.'
            (t.value64() as List<Float>) == expected

        when : 'The same underlying data is being queried by calling the device...'
            def result = (0..<t.size()).collect{device.valueFor(t, it)}

        then : 'This new result also contains the same elements.'
            result == expected

        where : 'The following data is being used for tensor instantiation :'
            device                | shape           | data                                               || expected
            Device.find("cpu")    | new int[]{3, 2} | new double[]{-5.0, -2.0, 1.0, -12.0, 3.0, -2.0}    || [-5.0, -2.0, 1.0, -12.0, 3.0, -2.0]
            Device.find("cpu")    | new int[]{3, 2} | new double[]{-1.0, -1.0, -1.0, 80.0, 3.0, -2.0}    || [-1.0, -1.0, -1.0, 80.0, 3.0, -2.0]

            Device.find("openCL") | new int[]{3, 2} | new double[]{-5.0, -2.0, 1.0, -12.0, 3.0, -2.0}    || [-5.0, -2.0, 1.0, -12.0, 3.0, -2.0]
            Device.find("openCL") | new int[]{3, 2} | new double[]{-1.0, -1.0, -1.0, 80.0, 3.0, -2.0}    || [-1.0, -1.0, -1.0, 80.0, 3.0, -2.0]
    }

    /**
     *  Every argument within an ExecutionCall instance has a purpose. Null is not permissible.
     */
    def 'Execution calls containing null arguments will cause an exception to be thrown in device instances.'(
        Device device
    ) {

        given : 'The given device is available and Neureka is being reset.'
            if ( device == null ) return
        and : 'A mocked ExecutionCall with mocked operation implementation and a mocked drain instantiator lambda...'
            def call = Mock(ExecutionCall)
            def implementation = Mock(Algorithm)

        when : 'The call is being passed to the device for execution...'
            device.execute(call)

        then : '...the implementation is being accessed in order to access the mocked lambda...'
            1 * call.getAlgorithm() >> implementation
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



    /**
     *  Device implementations always also behave as storage units for tensors.
     */
    def 'Devices store tensors which can also be restored.'(
            Device device
    ) {

        given : 'The given device is available and Neureka is being reset.'
            if ( device == null ) return
        and : 'Two tensors which will be transferred later on...'
            int initialNumber = device.size()
            Tsr a = new Tsr([2, 3], ";)")
            Tsr b = new Tsr([3, 4], ":P")

        expect : 'The given device is initially empty.'
            device.isEmpty() == ( device.size() == 0 )
            !device.has( a )
            !device.has( b )

        when : 'The the first tensor is being passed to the device...'
            device.store( a )

        then : '...tensor "a" is now on the device.'
            !device.isEmpty()
            device.size() == initialNumber + 1
            device.has( a )
            !device.has( b )

        when : 'The the second tensor is being passed to the device...'
            device.store( b )

        then : '...tensor "b" is now also on the device.'
            !device.isEmpty()
            device.size() == initialNumber + 2
            device.has( a )
            device.has( b )

        when : 'They are being removed again...'
            device.free( a ).free( b )

        then : '...the device is empty again.'
            device.isEmpty() == ( initialNumber == 0 )
            device.size() == initialNumber
            !device.has( a )
            !device.has( b )

        where : 'The following Device instances are being tested :'
            device << [
                    HostCPU.instance(),
                    Device.find( "openCL" ),
                    FileDevice.instance( "build/test-can" )
            ]

    }


}