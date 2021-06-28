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

    def setup()
    {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'Tensors as String can be formatted on an entry based level.'()
    {
        given : 'A new tensor of rank 2 storing Strings:'
            Tsr t = Tsr.of([2, 3], DataType.of(String.class), (i, indices) -> {
                return ["sweet", "salty", "blue", "spinning", "confused", "shining"].get( (i + 17**i)%6 ) + ' ' +
                    ["Saitan", "Apple", "Tofu",  "Strawberry", "Almond", "Salad"].get( (i + 7**i)%6 )
            })

        expect: 'When we convert the tensor to a String via the flags "b" (cell bound) and "f" (formatted).'
            t.toString('bf') == "(2x3):[\n" +
                                    "   [ sal.., swe.., spi.. ],\n" +
                                    "   [ blu.., shi.., con.. ]\n" +
                                    "]\n"
        and: 'When additionally supplying the flag "p" (padding) then most String entries will be printed fully.'
            t.toString('bfp') == "(2x3):[\n" +
                    "   [   salty Apple  ,    sweet Tofu  , spinning Stra.. ],\n" +
                    "   [   blue Almond  ,  shining Salad , confused Saitan ]\n" +
                    "]\n"
        and: 'When supplying the "p" and "b" flags manually then the result is as expected:'
            t.toString([
                    (TsrAsString.Should.BE_CELL_BOUND) : true,
                    (TsrAsString.Should.HAVE_PADDING_OF) : 4
            ]) == "(2x3):[salty Ap.., sweet Tofu, spinning.., blue Alm.., shining .., confused..]"

    }

    def 'Tensors as String can be formatted depending on shape.'(
            String mode, List<Integer> shape, String expected
    ){
        given: 'Four tensors of various data types:'
            Tsr t1 = Tsr.of( Float.class, shape, -4..5 ).set( Tsr.of( shape, -7..3 ) )
            Tsr t2 = Tsr.of( shape, -4..5 ).set( Tsr.of( shape, -7..3 ) )
            Tsr t3 = Tsr.of( Integer.class, shape, -4..5 ).set( Tsr.of( shape, -7..3 ) )
            Tsr t4 = Tsr.of( Short.class, shape, -4..5 ).set( Tsr.of( shape, -7..3 ) )

        expect: 'The first tensor has the expected internals ans produces the correct String representation.'
            t1.toString(mode) == expected
            t1.dataType == DataType.of( Float.class )
            t1.data instanceof float[]
        and : 'The second tensor has the expected internals ans produces the correct String representation.'
            t2.toString(mode) == expected
            t2.dataType == DataType.of( Double.class )
            t2.data instanceof double[]
        and : 'The third tensor has the expected internals ans produces the correct String representation.'
            t3.toString(mode).replace(' ','') == expected.replace('.0','  ').replace(' ','')
            t3.dataType == DataType.of( Integer.class )
            t3.data instanceof int[]
        and : 'The fourth tensor has the expected internals ans produces the correct String representation.'
            t4.toString(mode).replace(' ','') == expected.replace('.0','  ').replace(' ','')
            t4.dataType == DataType.of( Short.class )
            t4.data instanceof short[]

        where : 'The print configurations codes "mode", a common shape and expected String representation will be supplied:'
           mode | shape     | expected
           "fap" | [2,3]    | "(2x3):[\n   [  -4.0 ,  -3.0 ,  -2.0  ],\n   [  -1.0 ,   0.0 ,   1.0  ]\n]\n"
           "fa"  | [2,3]    | "(2x3):[\n   [ -4.0, -3.0, -2.0 ],\n   [ -1.0, 0.0, 1.0 ]\n]\n"
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
            Tsr t = Tsr.of( 6 )
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
            Tsr t = Tsr.of( new int[]{ 2 }, 5 )
        expect : 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
        when : 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
        then : 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !t.is64() && !t.is32()
            t.dataType.getTypeClass() == Neureka.get().settings().dtype().defaultDataTypeClass
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
            Tsr t = Tsr.of(  DataType.of(I8.class ), new int[]{ 2 } )
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
