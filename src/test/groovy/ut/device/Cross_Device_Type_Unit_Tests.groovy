package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.backend.api.Algorithm
import neureka.backend.api.ExecutionCall
import neureka.calculus.internal.CalcUtil
import neureka.common.utility.DataConverter
import neureka.devices.Device
import neureka.devices.file.FileDevice
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import neureka.view.TsrStringSettings
import spock.lang.Ignore
import spock.lang.IgnoreIf
import spock.lang.Specification

import java.util.function.BiConsumer

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
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() && type == OpenCLDevice }) // We need to assure that this system supports OpenCL!
    def 'Querying for Device implementations works as expected.'(
            String query, Class type
    ) {
        when : 'The query is being passed to the "find" method...'
            var device = Device.find(query)

        then : 'The resulting Device variable has the expected type.'
            device.class == type

        where :
            query                       || type
            "cPu"                       || CPU
            "jVm"                       || CPU
            "natiVe"                    || CPU
            "Threaded"                  || CPU
            "openCl"                    || OpenCLDevice
            "nvidia or amd or intel"    || OpenCLDevice // This assumes that there is an amd/intel/nvidia gpu!
            "first"                     || OpenCLDevice
    }


    /**
     * The data of a tensor located on an Device should
     * be update when passing a float or double array!
     */
    def 'Passing a numeric array to a tensor should modify its content!'(
            Device device, Object data1, Object data2, String expected
    ) {
        given : 'A 2D tensor is being instantiated..'
            Tsr t = Tsr.of(new int[]{3, 2}, new double[]{2, 4, -5, 8, 3, -2}).to(device)

        when : 'A numeric array is passed to said tensor...'
            t.setValue(data1)
            t.setValue(data2)

        then : 'The tensor (as String) contains the expected String.'
            t.toString().contains(expected)

        where : 'The following data is being used :'
            device                | data1                      | data2                      || expected
            Device.find("cpu")    | new float[0]               | new float[0]               || "(3x2):[2.0, 4.0, -5.0, 8.0, 3.0, -2.0]"
            Device.find("cpu")    | new float[]{2, 3, 4, 5, 6} | new float[]{1, 1, 1, 1}    || "(3x2):[1.0, 1.0, 1.0, 1.0, 6.0, -2.0]"
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
            Tsr t = Tsr.of(shape, data).to(device)

        then : 'The tensor values (as List) are as expected.'
            Arrays.equals(t.getValueAs(double[].class), DataConverter.get().convert(expected,double[].class))

        when : 'The same underlying data is being queried by calling the device...'
            var result = (0..<t.size()).collect{device.access(t).readAt(it)}

        then : 'This other result also contains the same elements.'
            result == expected

        when :
            result = (0..<t.size()).collect{device.access(t).readArray(data.getClass(), it, 1)[0]}
        then : 'This other result also contains the same elements.'
            result == expected


        where : 'The following data is being used for tensor instantiation :'
            device                | shape           | data                                               || expected
            Device.find("cpu")    | new int[]{3, 2} | new double[]{-5.0, -2.0, 1.0, -12.0, 3.0, -2.0}    || [-5.0, -2.0, 1.0, -12.0, 3.0, -2.0]

            Device.find("cpu")    | new int[]{3, 2} | new int[]{-5, -2, 1, -12, 3, -2}                   || [-5, -2, 1, -12, 3, -2]

            Device.find("cpu")    | new int[]{3, 2} | new long[]{-5, -2, 1, -12, 3, -2}                  || [-5, -2, 1, -12, 3, -2]

            Device.find("cpu")    | new int[]{3, 2} | new byte[]{-5, -2, 1, -12, 3, -2}                  || [-5, -2, 1, -12, 3, -2]

            Device.find("cpu")    | new int[]{3, 2} | new short[]{-5, -2, 1, -12, 3, -2}                 || [-5, -2, 1, -12, 3, -2]

            Device.find("cpu")    | new int[]{3, 2} | new float[]{-5, -2, 1, -12, 3, -2}                 || [-5, -2, 1, -12, 3, -2]

            Device.find("cpu")    | new int[]{3, 2} | new boolean[]{true, false, false}                  || [true, false, false, true, false, false]

            Device.find("openCL") | new int[]{3, 2} | new double[]{-5.0, -2.0, 1.0, -12.0, 3.0, -2.0}    || [-5.0, -2.0, 1.0, -12.0, 3.0, -2.0]
            Device.find("openCL") | new int[]{3, 2} | new float[]{-1.0, -1.0, -1.0, 80.0, 3.0, -2.0}     || [-1.0, -1.0, -1.0, 80.0, 3.0, -2.0]
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
            var call = Mock(ExecutionCall)
            var implementation = Mock(Algorithm)
        and :
            call.getDevice() >> device

        when : 'The call is being passed to the device for execution...'
            CalcUtil.recursiveExecution(call, (executionCall, executor) -> null)

        then : '...the implementation is being accessed in order to access the mocked lambda...'
            (1.._) * call.getAlgorithm() >> implementation
            1 * implementation.prepare(call) >> call
        and : 'The tensor array is being accessed to check for null. (For exception throwing)'
            1 * call.inputs() >> new Tsr[]{ Mock(Tsr), null }
        and : 'The expected exception is being thrown alongside a descriptive message.'
            def exception = thrown(IllegalArgumentException)
            exception.message == "Device arguments may not be null!\n" +
                    "One or more tensor arguments within the given ExecutionCall instance is null."

        where : 'The following Device instances are being tested :'
            device << [
                    CPU.get(),
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
            Tsr a = Tsr.of([2, 3], ";)")
            Tsr b = Tsr.of([3, 4], ":P")

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
                    CPU.get(),
                    Device.find( "openCL" ),
                    FileDevice.at( "build/test-can" )
            ]

    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() && (data.device instanceof OpenCLDevice) })
    def 'Devices store slices which can also be restored.'(
            Device device
    ) {
        given : 'The given device is available and Neureka is being reset.'
            if ( device == null ) return
        and : 'Two tensors which will be transferred later on...'
            int initialNumber = device.size()
            Tsr a = Tsr.of([2, 3], ";)")
            Tsr b = a[1, 0..2]

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
        and :
            !device.has( b )

        when :
            device.free( a )

        then : '...the device is empty again.'
            device.isEmpty() == ( initialNumber == 0 )
            device.size() == initialNumber
            !device.has( a )
            !device.has( b )

        where : 'The following Device instances are being tested :'
        device << [
                CPU.get(),
                Device.find( "openCL" )
        ]

    }

    @Ignore
    def 'Devices cannot store slices which parents are not already stored.'(
            Device device, BiConsumer<Device, Tsr> storageMethod
    ) {
        given : 'The given device is available and Neureka is being reset.'
            if ( device == null ) return
        and : 'Two tensors which will be transferred later on...'
            Tsr a = Tsr.of([2, 3], ";)")
            Tsr b = a[1, 0..2]
        and :
            var initialSize = device.size()

        expect : 'The given device is initially empty.'
            device.isEmpty() == ( device.size() == 0 )
            !device.has( a )
            !device.has( b )

        when : 'The the first tensor is being passed to the device...'
            device.store( b )

        then : '...tensor "a" is now on the device.'
            var exception = thrown(IllegalStateException)
            exception.message.contains("Data parent is not outsourced!")

        expect : 'The given device is initially empty.'
            device.isEmpty() == ( device.size() == 0 )
            !device.has( a )
            !device.has( b )

        when :
            storageMethod(device, a)

        then :
            !device.isEmpty()
            a.isOutsourced()
            b.isOutsourced()
        and :
            device.has( a )
            b.unsafe.data == null
            device.has( b )
            device.size() == initialSize

        where : 'The following Device instances are being tested :'
            device                                  | storageMethod
            Device.find( "openCL" )                 | { d, t -> d.store(t) }
            Device.find( "openCL" )                 | { d, t -> t.to(d)   }
            //FileDevice.at( "build/test-can" )     | { d, t -> d.store(t) }
            //CPU.get()                    | { d, t -> t.set(d)   }
            //FileDevice.at( "build/test-can" )     | { d, t -> t.set(d)   }
            //CPU.get()                    | { d, t -> d.store(t) }
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCL() && data.deviceType == 'GPU' })
    def 'Virtual tensors stay virtual when outsourced.'(
            String deviceType
    ) {
        given : 'We create a homogeneously filled tensor, which is therefor "virtual".'
            var t = Tsr.ofFloats().withShape(4,3).all(-0.54f)
        and : 'We also get a device for testing...'
            var device = Device.find(deviceType)

        expect : 'We expect that the tensor is virtual, meaning its underlying data array stores only a single value...'
            t.isVirtual()

        when : 'We send the tensor to the device...'
            t.to(device)
        then : 'This should cause it to be "outsourced", (except dor a CPU device of course).'
            t.isOutsourced() != ( device instanceof CPU )
        and : '...we expect the tensor to stay virtual on the device!'
            t.isVirtual()

        when : 'We restore the device...'
            device.restore(t)
        then : 'The tensor should no longer be outsourced.'
            !t.isOutsourced()
        and : 'It should still be virtual!'
            t.isVirtual()

        where : 'We test on the following devices:'
            deviceType << ['CPU','GPU']
    }


}