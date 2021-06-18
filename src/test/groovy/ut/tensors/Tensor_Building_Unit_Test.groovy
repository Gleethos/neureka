package ut.tensors

import neureka.Tsr
import neureka.dtype.DataType
import spock.lang.Specification

class Tensor_Building_Unit_Test extends Specification
{

    def 'Tensors can be created fluently.'(
            Class<Object> type, Number value, Object data
    ) {

        given : 'We create a new homogeneously filled Tsr instance using the fluent builder API.'
            Tsr<?> t = Tsr.of( type )
                                 .withShape( 3, 2 )
                                 .all( value )

        expect : 'This new instance will have the expected data type...'
            t.dataType == DataType.of(type)

        and : '...also it will contain the expected data.'
            t.data == data

        and : 'The tensor will have the shape we passed to the builder.'
            t.shape() == [3, 2]

        and : 'The size of the tensor will be the product of all shape entries!'
            t.size() == 6

        and : """
                Based on the fact that the tensor is homogeneously filled it will be a "virtual tensor"
                This means that the tensor will not have allocated the memory proportional to the size
                of the tensor!
            """
            t.isVirtual()

        where : 'The following data is being used to populate the builder API:'
            type          | value            || data
            Integer.class |   4   as int     || new int[]{4}
            Double.class  |   4.0 as double  || new double[]{4.0}
            Float.class   |   4f  as float   || new float[]{4f}

    }



    def 'Range based tensors can be created fluently.'(
            Class<Object> type, Number from, Number to, double step, Object data
    ) {

        given : 'We create a range based Tsr instance using the fluent builder API.'
            Tsr<?> t = Tsr.of( type )
                            .withShape( 3, 2 )
                            .iterativelyFilledFrom( from ).to( to ).step( step )

        expect : 'This new instance will have the expected data type...'
            t.dataType == DataType.of(type)

        and : '...also it will contain the expected data.'
            t.data == data

        and : 'The tensor will have the shape we passed to the builder.'
            t.shape() == [3, 2]

        and : 'The size of the tensor will be the product of all shape entries!'
            t.size() == 6

        and : """
                Based on the fact that the tensor is not homogeneously filled it will be not be a "virtual tensor".
                Virtual would mean that a tensor does not have allocated memory proportional to the size
                of the tensor! 
                In this case the tensor should be actual.
            """
            !t.isVirtual()

        where : 'The following data is being used to populate the builder API:'
        type          | from            | to                |  step  || data
        Integer.class | -9  as int      | 18  as int        |   2    || [-9, -7, -5, -3, -1, 1] as int[]
        Double.class  | 2.7 as double   | 45.0 as double    |   3    || [2.7, 5.7, 8.7, 11.7, 14.7, 17.7] as double[]
        Double.class  | -3 as double    | 3 as double       |   0.5  || [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5] as double[]
        Float.class   | 6.4f as float   | 78.3f  as float   |   4    || [6.4, 10.4, 14.4, 18.4, 22.4, 26.4] as float[]

    }



    def 'Value based tensors can be created fluently.'(
            Class<Object> type, Object[] data, Object expected
    ) {

        given : 'We create a range based Tsr instance using the fluent builder API.'
            Tsr<?> t = Tsr.of( type )
                .withShape( 3, 2 )
                .iterativelyFilledBy( data )

        expect : 'This new instance will have the expected data type...'
            t.dataType == DataType.of(type)

        and : '...also it will contain the expected data.'
            t.data == expected

        and : 'The tensor will have the shape we passed to the builder.'
            t.shape() == [3, 2]

        and : 'The size of the tensor will be the product of all shape entries!'
            t.size() == 6

        and : """
                Based on the fact that the tensor is not homogeneously filled it will be not be a "virtual tensor".
                Virtual would mean that a tensor does not have allocated memory proportional to the size
                of the tensor! 
                In this case the tensor should be actual.
            """
            !t.isVirtual()

        where : 'The following data is being used to populate the builder API:'
            type          | data                             || expected
            Integer.class | [2, 3, 4]           as Integer[] || [2, 3, 4, 2, 3, 4]                         as int[]
            Double.class  | [-5, 6.5, 7]        as Double[]  || [-5, 6.5, 7, -5, 6.5, 7]                   as double[]
            Short.class   | [6,  -1, -2]        as Short[]   || [6,  -1, -2, 6,  -1, -2]                   as short[]
            Float.class   | [22.4, 26.4]        as Float[]   || [22.4, 26.4, 22.4, 26.4, 22.4, 26.4]       as float[]
            //Byte.class    | [-20, 3, 4 -3]      as Byte[]    || [-20, 3, 4, -3, -20, 3]                    as byte[]
            Long.class    | [23, 199]           as Long[]    || [23, 199, 23, 199, 23, 199]                as long[]

    }




}
