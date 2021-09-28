package ut.utility

import neureka.utility.DataConverter
import spock.lang.Specification

class DataConverter_Spec extends Specification
{
    def 'The DataConverter can convert the given array data.'() {

        given :
            def converter = DataConverter.instance()
            def f32Array = new float[2]
            def f64Array = new double[2]
        expect :
            converter.convert( [-50, 2, 190] as byte[], BigInteger[].class ) == [-50, 2, -66] as BigInteger[]
            converter.convert( [-50, 2, 190] as byte[], short[].class ) == [-50, 2, -66] as short[]
            converter.convert( [-50, 2, 190] as byte[], int[].class ) == [-50, 2, -66] as int[]
            converter.convert( [-50, 2, 190] as byte[], long[].class ) == [-50, 2, -66] as long[]
        and :
            DataConverter.Utility.seededFloatArray( f32Array, 42 ) == [1.1419053, 0.91940796] as float[]
            DataConverter.Utility.newSeededFloatArray( 42, 2 ) == f32Array
            DataConverter.Utility.seededFloatArray( f32Array, "I'm a seed!") == [-0.037259135, -0.7227145] as float[]
            DataConverter.Utility.newSeededFloatArray( "I'm a seed!", 2 ) == f32Array
        and :
            DataConverter.Utility.seededDoubleArray( f64Array, 42 ) == [1.1419053154730547, 0.9194079489827879] as double[]
            DataConverter.Utility.newSeededDoubleArray( 42, 2 ) == f64Array
            DataConverter.Utility.seededDoubleArray( f64Array, "I'm a seed!") == [-0.03725913496921719, -0.722714495437272] as double[]
            DataConverter.Utility.newSeededDoubleArray( "I'm a seed!", 2 ) == f64Array

    }

}
