package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.opencl.CLContext
import neureka.devices.opencl.Data
import neureka.dtype.DataType
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Devices manage the states of the tensors they store!")
@Narrative('''

    Tensors should not manage their states
    themselves, simply because the type and location
    of the data is dependent on the device onto which they are stored.
    This specification tests of various device implementations
    enable reading to or writing from the tensors they store.

''')
class Cross_Device_IO_Spec extends Specification
{

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.deviceType == 'GPU' })
    def 'We can use the access device API to read from a tensor.'(
            String deviceType, Class<Object> type, Object[] fill, Object expected
    ) {
        given : 'We fetch the required device instance from its interface.'
            var device = Device.get(deviceType)
        and : 'We fetch the array type of the tested data type!'
            var arrayType = DataType.of(type).dataArrayType()
        and : 'A tensor filled with 4 values which we are going to store on the previously fetched device.'
            var t = Tsr.of(type).withShape(4).andFill(fill).to(device)
        and : 'A slice from the above tensor.'
            var s = t[1..2]

        expect : 'The device reports that both tensors have the same data array size.'
            device.access(t).dataSize == 4
            device.access(s).dataSize == 4
        and : 'Reading the underlying tensor data from the device will yield the expected result.'
            device.access(t).readAll(false) == expected
            device.access(s).readAll(false) == expected
            device.access(t).readAll(true)  == expected
            device.access(s).readAll(true)  == expected
        and : 'This is also true for when reading at a particular index.'
            device.access(t).readArray(arrayType, 1, 1) == [expected[1]]
            device.access(s).readArray(arrayType, 1, 1) == [expected[1]]
            device.access(t).readArray(arrayType, 2, 1) == [expected[2]]
            device.access(s).readArray(arrayType, 2, 1) == [expected[2]]

        where : 'We use the following data:'
            deviceType | type      | fill     ||  expected
            'CPU'      | Integer   | [2, 1]   || [2, 1, 2, 1]
            'CPU'      | Short     | [2,7,8]  || [2,7,8,2]
            'CPU'      | Long      | [6,2,6]  || [6,2,6,6]
            'CPU'      | Byte      | [6,2,7]  || [6,2,7,6]
            'CPU'      | Double    | [3.4, 3] || [3.4d, 3.0d, 3.4d, 3.0d]
            'CPU'      | Float     | [5.7,-1] || [5.7f, -1.0f, 5.7f, -1.0f]
            'GPU'      | Float     | [5.7,-1] || [5.7f, -1.0f, 5.7f, -1.0f]
            'CPU'      | Character | ['6','a']|| ['6','a','6','a']
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.deviceType == 'GPU' })
    def 'We can use the access device API to write to a tensor'(
            String deviceType, Class<Object> type, Object[] fill, Object write, Object expected
    ) {
        given : 'We fetch the required device instance from its interface.'
            var device = Device.get(deviceType)
        and : 'We fetch the array type of the tested data type!'
            var arrayType = DataType.of(type).dataArrayType()
        and : 'A tensor filled with 4 values which we are going to store on the previously fetched device.'
            var t = Tsr.of(type).withShape(4).andFill(fill).to(device)
        and : 'A slice from the above tensor.'
            var s = t[1..2]

        when : 'We write some test data into different ranges within the 2 tensors.'
            device.access(t).writeFrom(write, 0).intoRange(0,2)
            device.access(s).writeFrom(write, 0).intoRange(2,4)
        then : 'Reading the previously written data should yield the expected result.'
            device.access(t).readArray(arrayType, 0, 2) == [expected[0],expected[1]]
            device.access(s).readArray(arrayType, 2, 2) == [expected[2],expected[3]]

        where :
            deviceType | type     | fill     | write ||  expected
            'CPU'      | Integer  | [2, 1]   | [5]   || [5, 1, 5, 1]
            'CPU'      | Short    | [2,7,8]  | [7]   || [7,7,7,2]
            'CPU'      | Long     | [6,2,6]  | [1]   || [1,2,1,6]
            'CPU'      | Byte     | [6,2,7]  | [8]   || [8,2,8,6]
            'CPU'      | Double   | [3.4, 3] | [3]   || [3d, 3.0d, 3d, 3.0d]
            'CPU'      | Float    | [5.7,-1] | [4]   || [4f, -1.0f, 4f, -1.0f]
            'GPU'      | Float    | [5.7,-1] | [8]   || [8f, -1.0f, 8f, -1.0f]
    }


    def 'The OpenCLDevice specific Data class represents JVM data for OpenCL.'(
            Object data, int start, int size, Class<?> expectedType, List<?> expected
    ) {
        given : 'We make sure that any data will not automatically be converted to floats!'
            if ( Neureka.get().backend.has(CLContext) )
                Neureka.get().backend.get(CLContext).settings.autoConvertToFloat = false
        and : 'We create 2 different data objects, a full and a partial/sliced array.'
            var full = Data.of(data)
            var slice = Data.of(data, start, size)
        and : 'An expected array based on the previous slice indices!'
            var expected2 = expected[start..(start+size-1)]

        expect: 'Both data objects report the expected array types!'
            full.data.class == expectedType
            slice.data.class == expectedType
        and : 'Also they report the expected data array size.'
            full.length == expected.size()
            slice.length == expected2.size()
        and :
            full.data !== slice.data
            full.data == expected
            (data instanceof Number && slice.data == [data] ) || slice.data == expected2
        and : 'They produce OpenCL specific pointer objects.'
            full.pointer != null
            slice.pointer != null

        cleanup :
            if ( Neureka.get().backend.has(CLContext) )
                Neureka.get().backend.get(CLContext).settings.autoConvertToFloat = true

        where :
            data                   | start | size || expectedType | expected
            [2, 3, 6] as float[]   | 1     | 2    || float[]      | [2, 3, 6]
            [8,-2,5,-1] as float[] | 1     | 3    || float[]      | [8,-2,5,-1]
            4 as Float             | 0     | 1    || float[]      | [4]
            [2, 3, 6] as double[]  | 1     | 2    || double[]     | [2, 3, 6]
            [8,-2,5,-1] as double[]| 1     | 3    || double[]     | [8,-2,5,-1]
            4 as Double            | 0     | 1    || double[]     | [4]
            [2, 3, 6] as int[]     | 1     | 2    || int[]        | [2, 3, 6]
            [8,-2,5,-1] as int[]   | 1     | 3    || int[]        | [8,-2,5,-1]
            4 as Integer           | 0     | 1    || int[]        | [4]
            [2, 3, 6] as short[]   | 1     | 2    || short[]      | [2, 3, 6]
            [8,-2,5,-1] as short[] | 1     | 3    || short[]      | [8,-2,5,-1]
            4 as Short             | 0     | 1    || short[]      | [4]
            [2, 3, 6] as long[]    | 1     | 2    || long[]       | [2, 3, 6]
            [8,-2,5,-1] as long[]  | 1     | 3    || long[]       | [8,-2,5,-1]
            4 as Long              | 0     | 1    || long[]       | [4]
            [2, 3, 6] as byte[]    | 1     | 2    || byte[]       | [2, 3, 6]
            [8,-2,5,-1] as byte[]  | 1     | 3    || byte[]       | [8,-2,5,-1]
            4 as Byte              | 0     | 1    || byte[]       | [4]
    }

}
