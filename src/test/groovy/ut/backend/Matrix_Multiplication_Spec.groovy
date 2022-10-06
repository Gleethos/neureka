package ut.backend

import neureka.Tsr
import neureka.ndim.config.NDConfiguration
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Matrix Multiplication")
@Narrative('''
    
    The tensor API exposes a useful method for Matrix Multiplication.
    This specification not only demonstrates how to use this method
    but also shows how matrix multiplication work 
    for tensors with both row and column major layouts.
    (typically, column major is faster)
    
''')
class Matrix_Multiplication_Spec extends Specification
{
    def 'The "matMul" method allows us to perform matrix multiplication.'(Class<?> type)
    {
        given : 'We create 2 simple tensors with the shapes (1x7) and (7x8).'
            var a = Tsr.of(type).withShape(1,7).andFill(1, 2, 3)
            var b = Tsr.of(type).withShape(7,8).andFill(2, -4, -1)
        when : 'We perform matrix multiplication.'
            var c = a.matMul(b)
        then : 'The result is a tensor with the shape (1x8).'
            c.shape == [1, 8]
            c.items == [-22, -10, -7, -22, -10, -7, -22, -10]

        where : 'We can use the following data types:'
            type << [Integer, Long, Double, Float]
    }

    def 'The simple CPU matrix multiplication implementation works as expected.'(
            String layout, Class<Object> type, int M, int K, int N, Object A, Object B, def expectedC
    ) {
        reportInfo """
            Matrix multiplication is possible between matrices of various dimensions,
            data types as well as data layouts!
        """
        given : 'We instantiate 2 matrices based on the data from the data table at the end of this method.'
            Tsr<?> a = Tsr.of(type, [M,K] as int[], A)
            Tsr<?> b = Tsr.of(type, [K,N] as int[], B)
        and : 'We create the data layout type based on the provided string...'
            var dataLayout = layout == 'ROW' ? NDConfiguration.Layout.ROW_MAJOR : NDConfiguration.Layout.COLUMN_MAJOR
        and : 'After that we convert both matrices to the layout!'
            a.mut.toLayout( dataLayout )
            b.mut.toLayout( dataLayout )
        expect : 'This should of cause make report that they indeed have this new layout.'
            a.NDConf.layout == dataLayout
            b.NDConf.layout == dataLayout

        when : 'We now perform the matrix multiplication with the 2 matrix tensors...'
            Tsr<?> c = a.matMul(b)

        then : 'The result will have the expected (M x N) shape.'
            c.shape == [M,N]
        and : 'It should have the expected value array.'
            c.items == expectedC

        where : 'We use the following scenario parameters:'
           layout |  type  | M | K | N | A                        | B                              || expectedC
           'ROW'  | Double | 2 | 2 | 2 | [4, 3, 2, 1] as double[] | [-0.5, 1.5, 1, -2] as double[] || [1, 0, 0, 1 ] as double[]
           'ROW'  | Double | 1 | 2 | 1 | [-2,1] as double[]       | [-1, -1.5] as double[]         || [ 0.5 ] as double[]
           'ROW'  | Double | 2 | 1 | 2 | [-2,1] as double[]       | [-1, -1.5] as double[]         || [ 2.0, 3.0, -1.0, -1.5 ] as double[]
           'ROW'  | Float  | 2 | 2 | 2 | [4,3,2,1] as float[]     | [-0.5, 1.5, 1, -2] as float[]  || [ 1, 0, 0, 1 ] as float[]
           'ROW'  | Float  | 1 | 2 | 1 | [-2,1] as float[]        | [-1, -1.5] as float[]          || [ 0.5 ] as float[]
           'ROW'  | Float  | 2 | 1 | 2 | [-2,1] as float[]        | [-1, -1.5] as float[]          || [ 2.0, 3.0, -1.0, -1.5 ] as float[]
           'COL'  | Double | 2 | 2 | 2 | [4, 3, 2, 1] as double[] | [-0.5, 1.5, 1, -2] as double[] || [1, 0, 0, 1 ] as double[]
           'COL'  | Double | 1 | 2 | 1 | [-2,1] as double[]       | [-1, -1.5] as double[]         || [ 0.5 ] as double[]
           'COL'  | Double | 2 | 1 | 2 | [-2,1] as double[]       | [-1, -1.5] as double[]         || [ 2.0, 3.0, -1.0, -1.5 ] as double[]
           'COL'  | Float  | 2 | 2 | 2 | [4,3,2,1] as float[]     | [-0.5, 1.5, 1, -2] as float[]  || [ 1, 0, 0, 1 ] as float[]
           'COL'  | Float  | 1 | 2 | 1 | [-2,1] as float[]        | [-1, -1.5] as float[]          || [ 0.5 ] as float[]
           'COL'  | Float  | 2 | 1 | 2 | [-2,1] as float[]        | [-1, -1.5] as float[]          || [ 2.0, 3.0, -1.0, -1.5 ] as float[]
    }

}
