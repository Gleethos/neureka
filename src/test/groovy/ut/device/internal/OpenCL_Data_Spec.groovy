package ut.device.internal

import neureka.Neureka
import neureka.devices.opencl.CLBackend
import neureka.devices.opencl.JVMData
import spock.lang.Specification

class OpenCL_Data_Spec extends Specification
{
    def setup() {
        if ( Neureka.get().backend().has(CLBackend) )
            Neureka.get().backend().get(CLBackend).getSettings().autoConvertToFloat = false
    }

    def cleanup() {
        if ( Neureka.get().backend().has(CLBackend) )
            Neureka.get().backend().get(CLBackend).getSettings().autoConvertToFloat = true
    }


    def 'The OpenCLDevice specific Data class represents JVM data for OpenCL.'(
            Object data, int start, int size, Class<?> expectedType, List<?> expected
    ) {
        given : 'We make sure that any data will not automatically be converted to floats!'
            if ( Neureka.get().backend.has(CLBackend) )
                Neureka.get().backend.get(CLBackend).settings.autoConvertToFloat = false
        and : 'We create 2 different data objects, a full and a partial/sliced array.'
            var full = JVMData.of(data)
            var slice = JVMData.of(data, size, start)
        and : 'An expected array based on the previous slice indices!'
            var expected2 = expected[start..(start+size-1)]

        expect: 'Both data objects report the expected array types!'
            full.array.class == expectedType
            slice.array.class == expectedType
        and : 'Also they report the expected data array size.'
            full.length == expected.size()
            slice.length == expected2.size()
        and :
            full.array !== slice.array
            full.array == expected
            (data instanceof Number && slice.array == [data] ) || slice.array == expected2
        and : 'They produce OpenCL specific pointer objects.'
            full.pointer != null
            slice.pointer != null

        cleanup :
            if ( Neureka.get().backend.has(CLBackend) )
                Neureka.get().backend.get(CLBackend).settings.autoConvertToFloat = true

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


    def 'The "Data" class can represent various OpenCL data types.'(
      Object array, Class<Object> arrayType, Class<?> type,
      int itemSize, int size, int offset, String targetType
    ) {
        given :
            array = array.asType(arrayType)
            var data1 = JVMData.of(array, size, offset)
            var data2 = JVMData.of(type, size)
            var data3 = JVMData.of(array)

        expect :
            data1.array == array[offset..(offset+size-1)].asType(arrayType)
            data2.array == new int[size].asType(arrayType)
            data3.array == array
        and :
            data1.length == size
            data2.length == size
            data3.length == array.length
        and :
            data1.itemSize == itemSize
            data2.itemSize == itemSize
            data3.itemSize == itemSize
        and :
            data1.pointer != null
            data2.pointer != null
            data3.pointer != null
        and :
            data1.type.name() == targetType
            data2.type.name() == targetType
            data3.type.name() == targetType
        and :
            data1.array == (0..<data1.length).collect({it->data1.getElementAt((int)it)})
            data2.array == (0..<data2.length).collect({it->data2.getElementAt((int)it)})
            data3.array == (0..<data3.length).collect({it->data3.getElementAt((int)it)})

        where :
            array        | arrayType |  type    | itemSize | size | offset || targetType
            [1, 2]       | int[]     | Integer  |    4     | 1    | 1      ||  'I32'
            [8, 5, 2]    | int[]     | Integer  |    4     | 2    | 0      ||  'I32'
            [42]         | int[]     | Integer  |    4     | 1    | 0      ||  'I32'
            [1,2,3,4,5]  | int[]     | Integer  |    4     | 3    | 2      ||  'I32'

            [1, 2]       | byte[]    | Byte     |    1     | 1    | 1      ||  'I8'
            [8, 5, 2]    | byte[]    | Byte     |    1     | 2    | 0      ||  'I8'
            [42]         | byte[]    | Byte     |    1     | 1    | 0      ||  'I8'
            [1,2,3,4,5]  | byte[]    | Byte     |    1     | 3    | 2      ||  'I8'

            [1, 2]       | long[]    | Long     |    8     | 1    | 1      ||  'I64'
            [8, 5, 2]    | long[]    | Long     |    8     | 2    | 0      ||  'I64'
            [42]         | long[]    | Long     |    8     | 1    | 0      ||  'I64'
            [1,2,3,4,5]  | long[]    | Long     |    8     | 3    | 2      ||  'I64'

            [1, 2]       | short[]   | Short    |    2     | 1    | 1      ||  'I16'
            [8, 5, 2]    | short[]   | Short    |    2     | 2    | 0      ||  'I16'
            [42]         | short[]   | Short    |    2     | 1    | 0      ||  'I16'
            [1,2,3,4,5]  | short[]   | Short    |    2     | 3    | 2      ||  'I16'

            [0.3, -0.9]  | float[]   | Float    |    4     | 2    | 0      ||  'F32'
            [2, 0.5, 8]  | float[]   | Float    |    4     | 2    | 0      ||  'F32'

            [0.3, -0.9]  | double[]  | Double   |    8     | 2    | 0      ||  'F64'
            [2, 0.5, 8]  | double[]  | Double   |    8     | 2    | 0      ||  'F64'
            [0.6, 3, 0.2]| double[]  | Double   |    8     | 1    | 1      ||  'F64'
    }


}
