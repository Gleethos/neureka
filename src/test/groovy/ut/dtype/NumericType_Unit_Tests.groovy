package ut.dtype

import neureka.dtype.custom.F32
import neureka.dtype.custom.F64
import neureka.dtype.custom.I16
import neureka.dtype.custom.I32
import neureka.dtype.custom.I64
import neureka.dtype.custom.I8
import neureka.dtype.NumericType
import neureka.dtype.custom.UI16
import neureka.dtype.custom.UI32
import neureka.dtype.custom.UI64
import neureka.dtype.custom.UI8;
import spock.lang.Specification;

class NumericType_Unit_Tests extends Specification
{

    def setupSpec()
    {
        reportHeader """
            This specification covers implementations
            of the "NumericType" interface.
            Such classes are responsible for
            representing all numeric types including the ones
            which are foreign to the JVM, namely : 
            unsigned integer types.
        """
    }

    def 'NumericType implementations return their expected properties.'(
            NumericType type, int bytes, Class<?> target, Class<?> array, boolean signed
    ){

        expect : 'The type instance describes the expected number of bytes.'
            type.numberOfBytes() == bytes
        and : 'It describes the expected JVM target type.'
            type.targetType() == target
        and : 'It also describes the expected array type of said JVM target type.'
            type.targetArrayType() == array
        and : 'The instance knows if it is signed or not.'
            type.signed() == signed


        where : 'The following data is being used: '
            type        || bytes | target           | array                 | signed
            new I8()    || 1     | Byte.class       | byte[].class          | true
            new UI8()   || 1     | Short.class      | short[].class         | false
            new I16()   || 2     | Short.class      | short[].class         | true
            new UI16()  || 2     | Integer.class    | int[].class           | false
            new I32()   || 4     | Integer.class    | int[].class           | true
            new UI32()  || 4     | Long.class       | long[].class          | false
            new I64()   || 8     | Long.class       | long[].class          | true
            new UI64()  || 8     | BigInteger.class | BigInteger[].class    | false
            new F32()   || 4     | Float.class      | float[].class         | true
            new F64()   || 8     | Double.class     | double[].class        | true
    }


    def 'NumericType implementations behaive as expected.'(
            NumericType type, List<Byte> data, Number converted
    ){
        given :
            def result = type.convert(data as byte[])

        expect : 'The array of bytes  is being converted to a fitting JVM type.'
            result == converted
        //and : type.convert(result) == (data as byte[]) // TODO!

        where : 'The following NumericType instances and bytes are being used :'
            type      | data                         || converted
            new I8()  | [-23]                        || -23
            new UI8() | [-23]                        || 233

            new I16() | [2, 3]                       || new BigInteger(new byte[]{2, 3}).shortValueExact()
            new I16() | [-16, -53]                   || ((short)-3893)
            new I16() | [16, -53]                    || 4299
            new I16() | [-1, -1]                     || -1

            new UI16()| [2, 3]                       || new BigInteger(new byte[]{2, 3}).shortValueExact()
            new UI16()| [-16, -53]                   || (int)(0x10000 + ((short)-3893)) //:= https://stackoverflow.com/questions/7932701/read-byte-as-unsigned-short-java
            new UI16()| [16, -53]                    || 4299
            new UI16()| [-1, -1]                     || 65535

            new I32() | [22,-2, 3,-4]                || 385745916
            new I32() | [-22,-2, -3,-4]              || -352387588

            new UI32()| [22,-2, 3,-4]                || 385745916
            new UI32()| [-22,-2, -3,-4]              || 721354236

            new I64() | [99, 2, 1, 35, 2, 5, 37, 22] || 7134266009577661718
            new I64() | [-99, 2, 1, -35, 2,5,-37,22] || -7133136811068105962

            new UI64()| [99, 2, 1, 35, 2, 5, 37, 22] || 7134266009577661718
            new UI64()| [-99, 2, 1, -35, 2,5,-37,22] || 11313607262641445654
    }



}
