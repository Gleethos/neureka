package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.dtype.DataType
import neureka.dtype.custom.I8
import neureka.utility.TsrAsString
import spock.lang.Specification

class Tensor_State_Unit_Test extends Specification
{
    def setupSpec() {
        reportHeader """    
            <h2>Tensor State Tests</h2>
            <p>
                This unit test specification covers the expected state of newly instantiated tensors.
                Certain properties must have their expected default values.
            </p>
        """
    }

    def setup() {
        Neureka.instance().reset()
        // Configure printing of tensors to be more compact:
        Neureka.instance().settings().view().asString = TsrAsString.configFromCode("dgc")
    }

    def 'Tensors as String can be formatted depending on shape.'(
            String mode, List<Integer> shape, String expected
    ){

        given:
            Tsr t1 = new Tsr( shape, Float.class, -4..5 )
            Tsr t = new Tsr( shape, -4..5 ).set( new Tsr( shape, -7..3 ) )
            println(t.toString(mode))
            println(t.toString(mode).replace("\n", "\\n"))

        expect:
            t.toString(mode) == expected

        where :
           mode | shape     | expected
           "fap" | [2,3]    | "(2x3):[\n   [  -4.0 ,  -3.0 ,  -2.0  ],\n   [  -1.0 ,   0.0 ,   1.0  ]\n]\n"
           "fa"  | [2,3]    | "(2x3):[\n   [ -4.0, -3.0, -2.0 ],\n   [ -1.0, 0.0, 1.0 ]\n]\n"
           //"apg"| [2,3]    | "(2x3):[\n   [  -4.0 ,  -3.0 ,  -2.0  ],\n   [  -1.0 ,   0.0 ,   1.0  ]\n]\n"
           "fp" | [3,2]     | "(3x2):[\n   [  -4.0 ,  -3.0  ],\n   [  -2.0 ,  -1.0  ],\n   [   0.0 ,   1.0  ]\n]\n"
           "fp" | [2,3,4]   | "(2x3x4):[\n   [\n      [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ],\n      [   0.0 ,   1.0 ,   2.0 ,   3.0  ],\n      [   4.0 ,   5.0 ,  -4.0 ,  -3.0  ]\n   ],\n   [\n      [  -2.0 ,  -1.0 ,   0.0 ,   1.0  ],\n      [   2.0 ,   3.0 ,   4.0 ,   5.0  ],\n      [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ]\n   ]\n]\n"
           "fp" | [2,2,3,4] | "(2x2x3x4):[\n   [\n      [\n         [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ],\n         [   0.0 ,   1.0 ,   2.0 ,   3.0  ],\n         [   4.0 ,   5.0 ,  -4.0 ,  -3.0  ]\n      ],\n      [\n         [  -2.0 ,  -1.0 ,   0.0 ,   1.0  ],\n         [   2.0 ,   3.0 ,   4.0 ,   5.0  ],\n         [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ]\n      ]\n   ],\n   [\n      [\n         [   0.0 ,   1.0 ,   2.0 ,   3.0  ],\n         [   4.0 ,   5.0 ,  -4.0 ,  -3.0  ],\n         [  -2.0 ,  -1.0 ,   0.0 ,   1.0  ]\n      ],\n      [\n         [   2.0 ,   3.0 ,   4.0 ,   5.0  ],\n         [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ],\n         [   0.0 ,   1.0 ,   2.0 ,   3.0  ]\n      ]\n   ]\n]\n"
           "f"  | [2, 70]   | "(2x70):[\n   [ -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, ... 38 more ..., 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ],\n   [ -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, ... 38 more ..., 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ]\n]\n"
           "f"  | [2, 100]  | "(2x100):[\n   [ -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, ... 68 more ..., 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ],\n   [ -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, ... 68 more ..., 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ]\n]\n"
           "f"  | [70, 2]   | "(70x2):[\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   ... 38 more ...\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ]\n]\n"
    }

    def 'Newly instantiated and unmodified scalar tensor has expected state.'()
    {
        given: 'A new instance of a scalar tensor.'
            Tsr t = new Tsr( 6 )
        expect: 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
        and : 'The tensor contains the expected data.'
            t.value64() == [6] as double[]
            t.value32() == [6] as float[]
            t.data == [6] as double[]
            t.value == [6] as double[]

        when: 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
        then: 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !t.is64() && !t.is32()
            t.value64() == null
            t.value32() == null
            t.data == null
            t.value == null
        when: 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then: 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.value64() == [0] as double[]
            t.value32() == [0] as float[]
            t.data == [0] as double[]
            t.value == [0] as double[]
            t.isVirtual()

    }

    def 'Newly instantiated and unmodified vector tensor has expected state.'()
    {
        given : 'A new vector tensor is being instantiated.'
            Tsr t = new Tsr( new int[]{ 2 }, 5 )
        expect : 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
        when : 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
        then : 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !t.is64() && !t.is32()
            t.dataType.getTypeClass() == Neureka.instance().settings().dtype().defaultDataTypeClass
            t.value64() == null
            t.value32() == null
            t.data == null
            t.value == null
        when : 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then : 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.value64() == [0, 0] as double[]
            t.value32() == [0, 0] as float[]
            t.data == [0] as double[]
            t.value == [0, 0] as double[]
            t.isVirtual()
    }


    def 'Tensor created from shape and datatype has expected state.'()
    {
        given : 'A new vector tensor is being instantiated.'
            Tsr t = new Tsr( new int[]{ 2 }, DataType.of(I8.class ) )
        expect : 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
            t.value64() == [0, 0] as double[]
            t.value32() == [0, 0] as float[]
            t.data == [0] as byte[]
            t.value == [0, 0] as byte[]
            t.isVirtual()
        when : 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
        then : 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !t.is64() && !t.is32()
            t.dataType.getTypeClass() == I8.class
            t.value64() == null
            t.value32() == null
            t.data == null
            t.value == null
        when : 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then : 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.value64() == [0, 0] as double[]
            t.value32() == [0, 0] as float[]
            t.data == [0] as byte[]
            t.value == [0, 0] as byte[]
            t.isVirtual()
    }

}
