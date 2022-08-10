package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.dtype.DataType
import neureka.dtype.custom.I8
import neureka.view.NDPrintSettings
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("The Tensor Initialization and State Specification")
@Narrative('''
    
    This specification defines the expected states of freshly instantiated
    and initialized tensors.
    After a tensor was created successfully we expect it 
    to have certain properties like a shape, rank, type nnd data array
    among other things.
    
''')
class Tensor_State_Spec extends Specification
{
    def setupSpec() {
        reportHeader """    
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
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 15
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def 'A tensor can be instantiated from a target type and nested lists.'(
            Class<Object> type, List<Object> list, List<Integer> shape, Object expected
    ) {
        given : 'We instantiate a tensor based on a target type and a list of things.'
            var t = Tsr.of(type, list)

        expect : 'The tensor has the targeted type, shape and data array!'
            t.itemType == type
            t.shape() == shape
            t.unsafe.data == expected

        where :
            type   | list        || shape    | expected
            Double | [1,2,1]     || [3]      | [1, 2, 1] as double[]
            Float  | [5, -4]     || [2]      | [5, -4] as float[]
            Byte   | [3, 4]      || [2]      | [3, 4] as byte[]
            Byte   | [[3], [4]]  || [2, 1]   | [3, 4] as byte[]
            String | [['3', '4']]|| [1, 2]   | ['3', '4'] as String[]

    }

    def 'Tensors as String can be formatted on an entry based level.'()
    {
        given : 'A new tensor of rank 2 storing Strings:'
            Tsr t = Tsr.of(DataType.of(String.class), [2, 3], (i, indices) -> {
                return ["sweet", "salty", "blue", "spinning", "confused", "shining"].get( (i + 17**i)%6 ) + ' ' +
                       ["Saitan", "Apple", "Tofu",  "Strawberry", "Almond", "Salad"].get( (i + 7**i)%6  )
            })

        expect: 'When we convert the tensor to a String via the flags "b" (cell bound) and "f" (formatted).'
            t.toString({ NDPrintSettings it ->
                                    it.setHasSlimNumbers(false)
                                            .setIsScientific(true)
                                            .setIsCellBound(true)
                                            .setIsMultiline(true)
                                            .setCellSize(5)
                                })  == "(2x3):[\n" +
                                         "   [ sal.., swe.., spi.. ],\n" +
                                         "   [ blu.., shi.., con.. ]\n" +
                                         "]"
        and: 'When additionally supplying the flag "p" (padding) then most String entries will be printed fully.'
            t.toString({ NDPrintSettings it ->
                        it.setHasSlimNumbers(false)
                                .setIsScientific(true)
                                .setIsCellBound(true)
                                .setIsMultiline(true)
                                .setCellSize(15)
                    }) == "(2x3):[\n" +
                                       "   [   salty Apple  ,    sweet Tofu  , spinning Stra.. ],\n" +
                                       "   [   blue Almond  ,  shining Salad , confused Saitan ]\n" +
                                       "]"
        and: 'Whe can use a map of configuration configuration enums as keys and fitting objects as values:'
            t.toString(
                    { NDPrintSettings it -> it.setIsCellBound(true).setCellSize(10)}
            ) == "(2x3):[salty Ap.., sweet Tofu, spinning.., blue Alm.., shining .., confused..]"

        and: 'This way we can also configure a postfix and prefix as well as limit the number of entries in a row:'
            t.toString({ NDPrintSettings it ->
                                        it.setPrefix('START<|').setPostfix('|>END').setCellSize(0).setIsCellBound(false).setRowLimit(2) }
            ) == "START<|(2x3):[salty Apple, sweet Tofu, ... + 4 more]|>END"
    }


    def 'Numeric tensors as String can be formatted on an entry based level.'()
    {
        given : 'A new tensor of rank 2 storing floats:'
            Tsr t = Tsr.of(DataType.of(Float.class), [2, 3], (i, indices) -> (i%4)/3 as float )

        expect : 'When we convert the tensor to a String via the flag "f" (formatted).'
            t.toString(
                    { NDPrintSettings it -> it.setHasSlimNumbers(false).setIsScientific(true).setIsCellBound(false).setIsMultiline(false).setCellSize(3) }
            ) == "(2x3):[0.0, 0.33333, 0.66666, 1.0, 0.0, 0.33333]"
        and : 'Whe can use a map of configuration configuration enums as keys and fitting objects as values:'
            t.toString(
                { NDPrintSettings it -> it.setHasSlimNumbers(true).setIsScientific(true).setCellSize(0) }
                ) == "(2x3):[0, .33333, .66666, 1, 0, .33333]"

    }


    def 'Tensors as String can be formatted depending on shape.'(
            List<Integer> shape, String mode, String expected
    ){
        given: 'We configure a NDPrintSettings object.'
            def settings =
                    Neureka.get()
                            .settings()
                            .view()
                            .getNDPrintSettings()
                            .clone()
                            .setRowLimit(  mode.contains( "s" ) ? 3 : 32 )
                            .setIsScientific( mode.contains( "c" )  )
                            .setIsMultiline( mode.contains( "f" ) )
                            .setHasGradient( mode.contains( "g" ) )
                            .setCellSize(  mode.contains( "p" ) ? 6 : mode.contains( "f" ) ? 2 : 1  )
                            .setHasValue( !(mode.contains( "shp" ) || mode.contains("shape")) )
                            .setHasRecursiveGraph( mode.contains( "r" ) )
                            .setHasDerivatives(  mode.contains( "d" ) )
                            .setHasShape(  !mode.contains( "v" ) )
                            .setIsCellBound(  mode .contains( "b" ) )
                            .setPostfix(  "" )
                            .setPrefix(  "" )
                            .setHasSlimNumbers(  false )

        and : 'Four tensors of various data types:'
            Tsr<Float>   t1 = Tsr.of( Float.class,   shape, -4f..5f ).set( Tsr.of( shape, -7f..3f ) )
            Tsr<Double>  t2 = Tsr.of( Double.class,  shape, -4d..5d ).set( Tsr.of( shape, -7d..3d ) )
            Tsr<Integer> t3 = Tsr.of( Integer.class, shape, -4..5   ).set( Tsr.of( shape, -7..3   ) )
            Tsr<Short>   t4 = Tsr.of( Short.class,   shape, (-4 as short)..(5 as short) ).set( Tsr.of( shape, (-7 as short)..(3 as short) ) )
            Tsr<Byte>    t5 = Tsr.of( Byte.class,    shape, (-4 as byte )..(5 as byte ) ).set( Tsr.of( shape, (-7 as byte)..(3 as byte) ) )

        expect: 'The first tensor has the expected internals and produces the correct String representation.'
            t1.toString(settings) == expected
            t1.dataType == DataType.of( Float.class )
            t1.unsafe.data instanceof float[]
        and : 'The second tensor has the expected internals and produces the correct String representation.'
            t2.toString(settings) == expected
            t2.dataType == DataType.of( Double.class )
            t2.unsafe.data instanceof double[]
        and : 'The third tensor has the expected internals and produces the correct String representation.'
            t3.toString(settings).replace(' ','') == expected.replace('.0','  ').replace(' ','')
            t3.dataType == DataType.of( Integer.class )
            t3.unsafe.data instanceof int[]
        and : 'The fourth tensor has the expected internals and produces the correct String representation.'
            t4.toString(settings).replace(' ','') == expected.replace('.0','  ').replace(' ','')
            t4.dataType == DataType.of( Short.class )
            t4.unsafe.data instanceof short[]
        and : 'The fifth tensor has the expected internals and produces the correct String representation.'
            t5.toString(settings).replace(' ','') == expected.replace('.0','  ').replace(' ','')
            t5.dataType == DataType.of( Byte.class )
            t5.unsafe.data instanceof byte[]

        where : 'The print configurations codes "mode", a common shape and expected String representation will be supplied:'
            shape     | mode | expected
            [2,3]     | "fap"| "(2x3):[\n   [  -4.0 ,  -3.0 ,  -2.0  ],\n   [  -1.0 ,   0.0 ,   1.0  ]\n]"
            [2,3]     | "fa" | "(2x3):[\n   [ -4.0, -3.0, -2.0 ],\n   [ -1.0, 0.0, 1.0 ]\n]"
            [3,2]     | "fp" | "(3x2):[\n   [  -4.0 ,  -3.0  ],\n   [  -2.0 ,  -1.0  ],\n   [   0.0 ,   1.0  ]\n]"
            [2,3,4]   | "fp" | "(2x3x4):[\n   [\n      [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ],\n      [   0.0 ,   1.0 ,   2.0 ,   3.0  ],\n      [   4.0 ,   5.0 ,  -4.0 ,  -3.0  ]\n   ],\n   [\n      [  -2.0 ,  -1.0 ,   0.0 ,   1.0  ],\n      [   2.0 ,   3.0 ,   4.0 ,   5.0  ],\n      [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ]\n   ]\n]"
            [2,2,3,4] | "fp" | "(2x2x3x4):[\n   [\n      [\n         [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ],\n         [   0.0 ,   1.0 ,   2.0 ,   3.0  ],\n         [   4.0 ,   5.0 ,  -4.0 ,  -3.0  ]\n      ],\n      [\n         [  -2.0 ,  -1.0 ,   0.0 ,   1.0  ],\n         [   2.0 ,   3.0 ,   4.0 ,   5.0  ],\n         [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ]\n      ]\n   ],\n   [\n      [\n         [   0.0 ,   1.0 ,   2.0 ,   3.0  ],\n         [   4.0 ,   5.0 ,  -4.0 ,  -3.0  ],\n         [  -2.0 ,  -1.0 ,   0.0 ,   1.0  ]\n      ],\n      [\n         [   2.0 ,   3.0 ,   4.0 ,   5.0  ],\n         [  -4.0 ,  -3.0 ,  -2.0 ,  -1.0  ],\n         [   0.0 ,   1.0 ,   2.0 ,   3.0  ]\n      ]\n   ]\n]"
            [2, 70]   | "f"  | "(2x70):[\n   [ -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, ..38 more.., 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ],\n   [ -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, ..38 more.., 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ]\n]"
            [2, 100]  | "f"  | "(2x100):[\n   [ -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, ..68 more.., 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ],\n   [ -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, ..68 more.., 1.0, 2.0, 3.0, 4.0, 5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 ]\n]"
            [70, 2]   | "f"  | "(70x2):[\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   ... 38 more ...\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ],\n   [ -4.0, -3.0 ],\n   [ -2.0, -1.0 ],\n   [ 0.0, 1.0 ],\n   [ 2.0, 3.0 ],\n   [ 4.0, 5.0 ]\n]"
    }



    def 'Newly instantiated and unmodified scalar tensor has expected state.'()
    {
        given: 'A new instance of a scalar tensor.'
            Tsr t = Tsr.of( 6 )
        expect: 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
        and : 'The tensor contains the expected data.'
            t.getItemsAs( double[].class ) == [6] as double[]
            t.getItemsAs( float[].class  ) == [6] as float[]
            t.getDataAs( double[].class ) == [6] as double[]
            t.getDataAs( float[].class  ) == [6] as float[]
            t.unsafe.data == [6] as double[]
            t.items == [6] as double[]

        when: 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
            t.unsafe.setData(null)
        then: 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !(t.unsafe.data instanceof double[]) && !(t.unsafe.data instanceof float[])
            t.getItemsAs( double[].class ) == null
            t.getItemsAs( float[].class  ) == null
            t.getDataAs( double[].class ) == null
            t.getDataAs( float[].class  ) == null
            t.unsafe.data == null
            t.items == null
        when: 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then: 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.getItemsAs( double[].class ) == [0] as double[]
            t.getItemsAs( float[].class  ) == [0] as float[]
            t.getDataAs( double[].class ) == [0] as double[]
            t.getDataAs( float[].class  ) == [0] as float[]
            t.unsafe.data == [0] as double[]
            t.items == [0] as double[]
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
            t.unsafe.setData(null)
        then : 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !(t.unsafe.data instanceof double[]) && !(t.unsafe.data instanceof float[])
            t.dataType.getRepresentativeType() == Neureka.get().settings().dtype().defaultDataTypeClass
            t.getItemsAs( double[].class ) == null
            t.getItemsAs( float[].class ) == null
            t.unsafe.data == null
            t.items == null
        when : 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then : 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.getItemsAs( double[].class ) == [0, 0] as double[]
            t.getItemsAs( float[].class ) == [0, 0] as float[]
            t.unsafe.data == [0] as double[]
            t.items == [0, 0] as double[]
            t.isVirtual()
    }


    def 'Tensor created from shape and datatype has expected state.'()
    {
        given : 'A new vector tensor is being instantiated.'
            Tsr t = Tsr.of(  DataType.of(I8.class ), new int[]{ 2 } )
        expect : 'The tensor is not stored on another device, meaning that it is not "outsourced".'
            !t.isOutsourced()
            t.getItemsAs( double[].class ) == [0, 0] as double[]
            t.getItemsAs( float[].class ) == [0, 0] as float[]
            t.unsafe.data == [0] as byte[]
            t.items == [0, 0] as byte[]
            t.isVirtual()
        when : 'The flag "isOutsourced" is being set to false...'
            t.setIsOutsourced( true )
            t.unsafe.setData(null)
        then : 'The tensor is now outsourced and its data is gone. (garbage collected)'
            t.isOutsourced()
            !(t.unsafe.data instanceof double[]) && !(t.unsafe.data instanceof float[])
            t.dataType.getRepresentativeType() == I8.class
            t.getItemsAs( double[].class ) == null
            t.getItemsAs( float[].class ) == null
            t.unsafe.data == null
            t.items == null
        when : 'The "isOutsourced" flag is set to its original state...'
            t.setIsOutsourced( false )
        then : 'Internally the tensor reallocates an array of adequate size. (dependent on "isVirtual")'
            t.getItemsAs( double[].class ) == [0, 0] as double[]
            t.getItemsAs( float[].class ) == [0, 0] as float[]
            t.unsafe.data == [0] as byte[]
            t.items == [0, 0] as byte[]
            t.isVirtual()
    }

    @IgnoreIf({ data.device == 'GPU' && !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'The data and the value of a tensor a 2 different things!'(
        String device
    ) {
        given : 'We create a simple vector:'
            var v = Tsr.ofFloats().withShape(3).andFill(-2, 4, 8)
        and : 'And then we store it on the device we want to test.'
            v.to(device)

        when : 'We create a slice of the above vector, a scalar...'
            var s = v.slice().axis(0).at(1).get()

        then : 'The slice contains the expected value with respect to the slice parent...'
            v.at(1).get() == s.at(0).get()

        and : 'They both do not share the same value array.'
            v.items != s.items
        and : 'They so however share the same underlying data.'
            v.unsafe.data == s.unsafe.data
        and :
            s.items == [4]

        where : 'We test the following devices:'
            device << ['CPU']
    }


}
