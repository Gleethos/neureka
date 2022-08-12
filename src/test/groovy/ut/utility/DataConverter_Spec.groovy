package ut.utility


import neureka.common.utility.DataConverter
import spock.lang.Specification

class DataConverter_Spec extends Specification
{
    def 'The DataConverter can convert the given array data.'()
    {
        given :
            def converter = DataConverter.get()
        expect :
            converter.convert( [-50, 2, 190] as byte[], BigInteger[].class ) == [-50, 2, -66] as BigInteger[]
            converter.convert( [-50, 2, 190] as byte[], short[].class ) == [-50, 2, -66] as short[]
            converter.convert( [-50, 2, 190] as byte[], int[].class ) == [-50, 2, -66] as int[]
            converter.convert( [-50, 2, 190] as byte[], long[].class ) == [-50, 2, -66] as long[]
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
