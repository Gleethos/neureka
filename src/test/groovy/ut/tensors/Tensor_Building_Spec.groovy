package ut.tensors

import neureka.Tsr
import neureka.dtype.DataType
import neureka.ndim.Initializer
import spock.lang.Specification

class Tensor_Building_Spec extends Specification
{

    def 'Tensors can be created fluently.'(
            Class<Object> type, Object value, Object data
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
            type      | value          || data
            Integer   |  4   as int    || new int[]   { 4   }
            Double    |  4.0 as double || new double[]{ 4.0 }
            Float     |  4f  as float  || new float[] { 4f  }
            Long      |  42L as Long   || new long[]  { 42L }
            Boolean   |  false         || new boolean[] { false }
            Character | '°' as char    || new char[] { '°' as char }
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
                Based on the fact that the tensor is not homogeneously filled it will be an "actual tensor".
                The opposite of that, a "virtual tensor", would mean that a tensor does not have allocated 
                memory proportional to the size of the tensor! 
                In this case however the tensor should be actual which means that it is not virtual.
            """
            !t.isVirtual()

        where : 'The following data is being used to populate the builder API:'
            type          | from            | to                |  step  || data
            Integer.class | -9   as int     | 18     as int     |   2    || [-9, -7, -5, -3, -1, 1]              as int[]
            Integer.class | -2   as int     | 4      as int     |   2    || [-2, 0, 2, 4, -2, 0]                 as int[]
            Double.class  | 2.7  as double  | 45.0   as double  |   3    || [2.7, 5.7, 8.7, 11.7, 14.7, 17.7]    as double[]
            Double.class  | -3   as double  | 3      as double  |   0.5  || [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5] as double[]
            Float.class   | 6.4f as float   | 78.3f  as float   |   4    || [6.4, 10.4, 14.4, 18.4, 22.4, 26.4]  as float[]
            Float.class   | 0f   as float   | 1f     as float   |   0.2f || [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]       as float[]
            Byte.class    | -5   as byte    | 6      as byte    |   2    || [-5, -3, -1, 1, 3, 5]                as byte[]
            Long.class    | -65  as long    | 45     as long    |   5    || [-65, -60, -55, -50, -45, -40]       as long[]


    }



    def 'Value based tensors can be created fluently.'(
            Class<Object> type, Object data, Object expected
    ) {

        given : 'We create a Tsr instance by passing an array of arguments which ought to iteratively fill the instance.'
            Tsr<?> t = Tsr.of( type )
                            .withShape( 3, 2 )
                            .andFill( data )

        expect : 'This new instance will have the expected data type...'
            t.dataType == DataType.of(type)

        and : '...also it will contain the expected data.'
            t.data == expected

        and : 'The tensor will have the shape we passed to the builder.'
            t.shape() == [3, 2]

        and : 'The size of the tensor will be the product of all shape entries!'
            t.size() == 6

        and : """
                Based on the fact that the tensor is not homogeneously filled it will be an "actual tensor".
                The opposite of that, a "virtual tensor", would mean that a tensor does not have allocated 
                memory proportional to the size of the tensor! 
                In this case however the tensor should be actual which means that it is not virtual.
            """
            !t.isVirtual()

        where : 'The following data is being used to populate the builder API:'
            type            | data                           || expected
            Integer.class   | [2, 3, 4]       as Integer[]   || [2, 3, 4, 2, 3, 4]                      as int[]
            Double.class    | [-5, 6.5, 7]    as Double[]    || [-5, 6.5, 7, -5, 6.5, 7]                as double[]
            Short.class     | [6,  -1, -2]    as Short[]     || [6,  -1, -2, 6,  -1, -2]                as short[]
            Float.class     | [22.4, 26.4]    as Float[]     || [22.4, 26.4, 22.4, 26.4, 22.4, 26.4]    as float[]
            Byte.class      | [-20, 3, 4, -3] as Byte[]      || [-20, 3, 4, -3, -20, 3]                 as byte[]
            Long.class      | [23, 199]       as Long[]      || [23, 199, 23, 199, 23, 199]             as long[]
            Boolean.class   | [true, false]   as Boolean[]   || [true, false, true, false, true, false] as boolean[]
            Character.class | ['x', 'y']      as Character[] || ['x', 'y', 'x', 'y', 'x', 'y']          as char[]
    }


    def 'Initialization lambda based tensors can be created fluently.'(
            Class<Object> type, Initializer initializer, Object expected
    ) {
        given : 'We create a Tsr instance by passing an initialization lambda which ought to iteratively fill the instance.'
            Tsr<?> t = Tsr.of( type )
                            .withShape( 3, 2 )
                            .andWhere( initializer )

        expect : 'This new instance will have the expected data type...'
            t.dataType == DataType.of(type)

        and : '...also it will contain the expected data.'
            t.data == expected

        and : 'The tensor will have the shape we passed to the builder.'
            t.shape() == [3, 2]

        and : 'The size of the tensor will be the product of all shape entries!'
            t.size() == 6

        and : """
                Based on the fact that the tensor is not homogeneously filled it will be an "actual tensor".
                The opposite of that, a "virtual tensor", would mean that a tensor does not have allocated 
                memory proportional to the size of the tensor! 
                In this case however the tensor should be actual which means that it is not virtual.
            """
            !t.isVirtual()

        where : 'The following data is being used to populate the builder API:'
            type           | initializer                                    || expected
            Integer.class  | { i, indices ->          (i + indices.sum()) } || [0, 2, 3, 5, 6, 8] as int[]
            Double.class   | { i, indices -> (double) (i + indices.sum()) } || [0, 2, 3, 5, 6, 8] as double[]
            Short.class    | { i, indices -> (short)  (i + indices.sum()) } || [0, 2, 3, 5, 6, 8] as short[]
            Float.class    | { i, indices -> (float)  (i + indices.sum()) } || [0, 2, 3, 5, 6, 8] as float[]
            Byte.class     | { i, indices -> (byte)   (i + indices.sum()) } || [0, 2, 3, 5, 6, 8] as byte[]
            Long.class     | { i, indices -> (long)   (i + indices.sum()) } || [0, 2, 3, 5, 6, 8] as long[]
            Character.class| { i, indices -> (char)   (i + indices.sum()) } || [0, 2, 3, 5, 6, 8] as char[]
            Boolean.class  | { i, indices -> (boolean)(i % 2 == 0) }        || [true, false, true, false, true, false] as boolean[]
    }


    def 'Vectors can be created fluently.'(
            Class<Object> type, Object[] values, Object data
    ) {

        given : 'We create a new Tsr instance using the "vector" method in the fluent builder API.'
            Tsr<?> t = Tsr.of( type ).vector( values )

        expect : 'This new instance will have the expected data type...'
            t.dataType == DataType.of(type)

        and : '...also it will contain the expected data.'
            t.data == data

        and : 'The tensor will have a one dimensional shape of the same length as the provided data array.'
            t.shape() == [values.length]

        and : 'The size of the tensor will also be as long as the data array!'
            t.size() == values.length

        and :  """
                Based on the fact that the tensor is not homogeneously filled it will be an "actual tensor".
                The opposite of that, a "virtual tensor", would mean that a tensor does not have allocated 
                memory proportional to the size of the tensor! 
                In this case however the tensor should be actual which means that it is not virtual.
            """
            !t.isVirtual()

        where : 'The following data is being used to populate the builder API:'
            type          | values                    || data
            Integer.class | [4, 5, -2]   as Integer[] || new int[]   { 4, 5, -2   }
            Double.class  | [-1, 7.5]    as Double[]  || new double[]{ -1, 7.5    }
            Float.class   | [0.6, -32.7] as Float[]   || new float[] { 0.6, -32.7 }
            Long.class    | [1, 3, 2, 4] as Long[]    || new long[]  { 1, 3, 2, 4 }
    }


    def 'Scalars can be created fluently.'(
            Class<Object> type, Object value, Object data
    ) {

        given : 'We create a new Tsr instance using the "scalar" method in the fluent builder API.'
            Tsr<?> t = Tsr.of( type ).scalar( value )

        expect : 'This new instance will have the expected data type...'
            t.dataType == DataType.of(type)

        and : '...also it will contain the expected data.'
            t.data == data

        and : 'The tensor will have a one dimensional shape of 1.'
            t.shape() == [1]

        and : 'The size of the tensor will also 1!'
            t.size() == 1
        /*
        and :  """
                The tensor has allocated memory proportional to the size of the tensor, namely 1 to 1! 
                Therefore the tensor is an "actual tensor", which means that it is not virtual.
            """
            !t.isVirtual() // TODO: FIX
        */
        where : 'The following data is being used to populate the builder API:'
            type          | value           || data
            Integer.class | 3    as Integer || new int[]   { 3    }
            Double.class  | 5.7  as Double  || new double[]{ 5.7  }
            Float.class   | 9.4f as Float   || new float[] { 9.4f }
            Long.class    | 42L  as Long    || new long[]  { 42L  }
    }



}
