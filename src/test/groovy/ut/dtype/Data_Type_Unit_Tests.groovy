package ut.dtype

import neureka.dtype.custom.F32
import neureka.dtype.custom.F64
import neureka.dtype.custom.I16
import neureka.dtype.custom.I32
import neureka.dtype.custom.I64
import neureka.dtype.custom.I8
import neureka.dtype.NumericType
import neureka.dtype.custom.UI16
import neureka.dtype.custom.UI64
import neureka.dtype.custom.UI8;
import spock.lang.Specification;

class Data_Type_Unit_Tests extends Specification
{

    def 'NumericType implementations return their expected properties.'(
            NumericType type, int bytes, Class<?> target, Class<?> array
    ){

        expect : 'The type instance describes the expected number of bytes.'
            type.numberOfBytes() == bytes
        and : 'It describes the expected JVM target type.'
            type.targetType() == target
        and : 'It also describes the expected array type of said JVM target type.'
            type.targetArrayType() == array


        where : 'The following data is being used: '
            type        || bytes | target           | array
            new I8()    || 1     | Byte.class       | byte[].class
            new UI8()   || 1     | Short.class      | short[].class
            new I16()   || 2     | Short.class      | short[].class
            new UI16()  || 2     | Integer.class    | int[].class
            new I32()   || 4     | Integer.class    | int[].class
            new I64()   || 8     | Long.class       | long[].class
            new UI64()  || 8     | BigInteger.class | BigInteger[].class
            new F32()   || 4     | Float.class      | float[].class
            new F64()   || 8     | Double.class     | double[].class

    }



}
