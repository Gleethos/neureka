package ut.dtype

import neureka.dtype.custom.I16
import neureka.dtype.custom.I8
import neureka.dtype.custom.NumericType
import neureka.dtype.custom.UI16
import neureka.dtype.custom.UI8;
import spock.lang.Specification;

class Data_Type_Unit_Tests extends Specification
{

    def 'Test types.'(
            NumericType type, int bytes, Class<?> target
    ){

        expect :
            type.numberOfBytes() == bytes
            type.targetType() == target


        where :
            type        || bytes | target
            new I8()    || 1     | Byte.class
            new UI8()   || 1     | Short.class
            new I16()   || 2     | Short.class
            new UI16()  || 2     | Integer.class

    }



}
