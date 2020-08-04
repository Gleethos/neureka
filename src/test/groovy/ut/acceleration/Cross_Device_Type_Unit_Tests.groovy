package ut.acceleration

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import spock.lang.Specification


class Cross_Device_Type_Unit_Tests extends Specification
{
    /**
     * The data of a tensor located on an Device should
     * be update when passing a float or double array!
     */
    def 'Passing a numeric array to a tensor should modify its content!'(
            Object data1, Object data2, String expected
    ) {
        given :
            Neureka.instance().reset()
            Tsr t = new Tsr(new int[]{3, 2}, new double[]{2, 4, -5, 8, 3, -2})
        and :
            if( data1 instanceof float[] ) t.setValue32(data1)
            else t.setValue64(data1 as double[])
            if( data2 instanceof float[] ) t.setValue32(data2)
            else t.setValue64(data2 as double[])

        expect :
            t.toString().contains(expected)
            if( Neureka.instance().canAccessOpenCL() ){
                t.add(Device.find("nvidia"))
                t.toString().contains(expected)
            }

        where :
            data1                      | data2                      || expected
            new float[0]               | new float[0]               || "(3x2):[2.0, 4.0, -5.0, 8.0, 3.0, -2.0]"
            new float[]{2, 3, 4, 5, 6} | new float[]{1, 1, 1, 1, 1} || "(3x2):[1.0, 1.0, 1.0, 1.0, 1.0, -2.0]"
            new float[]{3, 5, 6}       | new float[]{4, 2, 3}       || "(3x2):[4.0, 2.0, 3.0, 8.0, 3.0, -2.0]"
            new double[]{9, 4, 7, -12} | new double[]{-5, -2, 1}    || "(3x2):[-5.0, -2.0, 1.0, -12.0, 3.0, -2.0]"
            new float[]{22, 24, 35, 80}| new double[]{-1, -1, -1}   || "(3x2):[-1.0, -1.0, -1.0, 80.0, 3.0, -2.0]"
    }


}