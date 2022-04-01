package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.dtype.DataType
import spock.lang.IgnoreIf
import spock.lang.Specification

class Cross_Device_IO_Spec extends Specification
{
    @IgnoreIf({ !Neureka.get().canAccessOpenCL() && deviceType == 'GPU' })
    def 'We can use the access device API to read from a tensor.'(
            String deviceType, Class<Object> type, Object[] fill, Object expected
    ) {
        given :
            var device = Device.find(deviceType)
            var arrayType = DataType.of(type).dataArrayType()
        and :
            var t = Tsr.of(type).withShape(4).andFill(fill).to(device)
        and :
            var s = t[1..2]

        expect :
            device.access(t).readAll(false) == expected
            device.access(s).readAll(false) == expected
            device.access(t).readAll(true)  == expected
            device.access(s).readAll(true)  == expected
        and :
            device.access(t).readArray(arrayType, 1, 1) == [expected[1]]
            device.access(s).readArray(arrayType, 1, 1) == [expected[1]]
            device.access(t).readArray(arrayType, 2, 1) == [expected[2]]
            device.access(s).readArray(arrayType, 2, 1) == [expected[2]]

        where :
            deviceType | type     | fill     ||  expected
            'CPU'      | Integer  | [2, 1]   || [2, 1, 2, 1]
            'CPU'      | Short    | [2,7,8]  || [2,7,8,2]
            'CPU'      | Long     | [6,2,6]  || [6,2,6,6]
            'CPU'      | Byte     | [6,2,7]  || [6,2,7,6]
            'CPU'      | Double   | [3.4, 3] || [3.4d, 3.0d, 3.4d, 3.0d]
            'CPU'      | Float    | [5.7,-1] || [5.7f, -1.0f, 5.7f, -1.0f]
            'GPU'      | Float    | [5.7,-1] || [5.7f, -1.0f, 5.7f, -1.0f]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() && deviceType == 'GPU' })
    def 'We can use the access device API to write to a tensor'(
            String deviceType, Class<Object> type, Object[] fill, Object write, Object expected
    ) {
        given :
            var device = Device.find(deviceType)
            var arrayType = DataType.of(type).dataArrayType()
        and :
            var t = Tsr.of(type).withShape(4).andFill(fill).to(device)
        and :
            var s = t[1..2]

        when :
            device.access(t).writeFrom(write, 0).intoRange(0,2)
            device.access(s).writeFrom(write, 0).intoRange(2,4)
        then :
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


}
