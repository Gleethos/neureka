package ut.utility

import neureka.backend.main.operations.other.Randomization
import neureka.common.utility.DataConverter
import spock.lang.Specification

class DataConverter_Spec extends Specification
{
    def 'The DataConverter can convert the given array data.'() {

        given :
            def converter = DataConverter.get()
            def f32Array = new float[2]
            def f64Array = new double[2]
        expect :
            converter.convert( [-50, 2, 190] as byte[], BigInteger[].class ) == [-50, 2, -66] as BigInteger[]
            converter.convert( [-50, 2, 190] as byte[], short[].class ) == [-50, 2, -66] as short[]
            converter.convert( [-50, 2, 190] as byte[], int[].class ) == [-50, 2, -66] as int[]
            converter.convert( [-50, 2, 190] as byte[], long[].class ) == [-50, 2, -66] as long[]
        and :
            Randomization.fillRandomly( f32Array, 42 ) == [-0.48528543, -1.4547276] as float[]
            Randomization.fillRandomly( new float[2], 42 ) == f32Array
            Randomization.fillRandomly( f32Array, "I'm a seed!") == [-0.36105144, 0.09591457] as float[]
            Randomization.fillRandomly( new float[2], "I'm a seed!" ) == f32Array
        and :
            Randomization.fillRandomly( f64Array, 42 ) == [-0.485285426832534, -1.454727674701364] as double[]
            Randomization.fillRandomly( new double[2], 42 ) == f64Array
            Randomization.fillRandomly( f64Array, "I'm a seed!") == [-0.36105142873347185, 0.09591457295459412] as double[]
            Randomization.fillRandomly( new double[2], "I'm a seed!" ) == f64Array

    }


    def 'An array of any type of object may be converted to a array of primitives.'() {

        expect :
            DataConverter.Utility.objectsToFloats([1, 2, 3].toArray(), 2) == [1, 2] as float[]
            DataConverter.Utility.objectsToDoubles([1, 2, 3].toArray(), 2) == [1, 2] as double[]
            DataConverter.Utility.objectsToBytes([1, 2, 3].toArray(), 2) == [1, 2] as byte[]
            DataConverter.Utility.objectsToInts([1, 2, 3].toArray(), 2) == [1, 2] as int[]
            DataConverter.Utility.objectsToLongs([1, 2, 3].toArray(), 2) == [1, 2] as long[]
            DataConverter.Utility.objectsToShorts([1, 2, 3].toArray(), 2) == [1, 2] as short[]
        and :
            DataConverter.Utility.objectsToFloats([2.2, 3.9].toArray(), 2) == [2.2,3.9] as float[]
            DataConverter.Utility.objectsToDoubles([2.2, 3.9].toArray(), 2) == [2.2, 3.9] as double[]
            DataConverter.Utility.objectsToBytes([2.2, 3.9].toArray(), 2) == [2, 3] as byte[]
            DataConverter.Utility.objectsToInts([2.2, 3.9].toArray(), 2) == [2, 3] as int[]
            DataConverter.Utility.objectsToLongs([2.2, 3.9].toArray(), 2) == [2, 3] as long[]
            DataConverter.Utility.objectsToShorts([2.2, 3.9].toArray(), 2) == [2, 3] as short[]


    }


}
