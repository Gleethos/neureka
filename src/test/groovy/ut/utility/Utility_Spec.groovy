package ut.utility

import neureka.common.utility.DataConverter
import spock.lang.Specification

class Utility_Spec extends Specification {


    def 'Object arrays can be converted to primitive arrays.'(
            Object[] input, def code, Class<?> expectedType
    ) {
        when :
            def result = code(input)

        then :
            result == input
        and :
            result.class != input.class
        and :
            result.class == expectedType

        where :
            input                    | code                                                 || expectedType
            [1, 2, 3] as Float[]     | {DataConverter.Utility.objFloatsToPrimFloats(it)}    || float[]
            [1, 2, 3] as Double[]    | {DataConverter.Utility.objDoublesToPrimDoubles(it)}  || double[]
            [1, 2, 3] as Integer[]   | {DataConverter.Utility.objIntsToPrimInts(it)}        || int[]
            [1, 2, 3] as Long[]      | {DataConverter.Utility.objLongsToPrimLongs(it)}      || long[]
            [1, 2, 3] as Short[]     | {DataConverter.Utility.objShortsToPrimShorts(it)}    || short[]
            [1, 2, 3] as Byte[]      | {DataConverter.Utility.objBytesToPrimBytes(it)}      || byte[]
            [1, 2, 3] as Boolean[]   | {DataConverter.Utility.objBooleansToPrimBooleans(it)}|| boolean[]
            [1, 2, 3] as Character[] | {DataConverter.Utility.objCharsToPrimChars(it)}      || char[]
    }


}
