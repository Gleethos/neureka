package it.backend

import neureka.Tsr
import neureka.ndim.config.NDConfiguration
import spock.lang.Specification

class Matrix_Multiplication_Spec extends Specification {

    def 'The simple CPU matrix multiplication implementation works as expected.'(
            String layout, Class<Object> type, int M, int K, int N, Object A, Object B, def expectedC
    ) {
        given :
            Tsr<?> a = Tsr.of(type, [M,K] as int[], A)
            Tsr<?> b = Tsr.of(type, [K,N] as int[], B)
        and :
            def dataLayout = layout == 'ROW' ? NDConfiguration.Layout.ROW_MAJOR : NDConfiguration.Layout.COLUMN_MAJOR
        and :
            a.unsafe.toLayout( dataLayout )
            b.unsafe.toLayout( dataLayout )
        expect :
            a.NDConf.layout == dataLayout
            b.NDConf.layout == dataLayout

        when :
            Tsr<?> c = a.matMul(b)

        then :
            c.shape() == [M,N]
        and :
            c.value == expectedC

        where :
           layout |  type  | M | K | N | A                     | B                              || expectedC
           'ROW'  | Double | 2 | 2 | 2 | [4, 3, 2, 1] as double[] | [-0.5, 1.5, 1, -2] as double[] || [1, 0, 0, 1 ] as double[]
           'ROW'  | Double | 1 | 2 | 1 | [-2,1] as double[]    | [-1, -1.5] as double[]         || [ 0.5 ] as double[]
           'ROW'  | Double | 2 | 1 | 2 | [-2,1] as double[]    | [-1, -1.5] as double[]         || [ 2.0, 3.0, -1.0, -1.5 ] as double[]
           'ROW'  | Float  | 2 | 2 | 2 | [4,3,2,1] as float[]  | [-0.5, 1.5, 1, -2] as float[]  || [ 1, 0, 0, 1 ] as float[]
           'ROW'  | Float  | 1 | 2 | 1 | [-2,1] as float[]     | [-1, -1.5] as float[]          || [ 0.5 ] as float[]
           'ROW'  | Float  | 2 | 1 | 2 | [-2,1] as float[]     | [-1, -1.5] as float[]          || [ 2.0, 3.0, -1.0, -1.5 ] as float[]
           //'COL'  | Double | 2 | 2 | 2 | [4, 3, 2, 1] as double[] | [-0.5, 1.5, 1, -2] as double[] || [1, 0, 0, 1 ] as double[]
           //'COL'  | Double | 1 | 2 | 1 | [-2,1] as double[]    | [-1, -1.5] as double[]         || [ 0.5 ] as double[]
           //'COL'  | Double | 2 | 1 | 2 | [-2,1] as double[]    | [-1, -1.5] as double[]         || [ 2.0, 3.0, -1.0, -1.5 ] as double[]
           //'COL'  | Float  | 2 | 2 | 2 | [4,3,2,1] as float[]  | [-0.5, 1.5, 1, -2] as float[]  || [ 1, 0, 0, 1 ] as float[]
           //'COL'  | Float  | 1 | 2 | 1 | [-2,1] as float[]     | [-1, -1.5] as float[]          || [ 0.5 ] as float[]
           //'COL'  | Float  | 2 | 1 | 2 | [-2,1] as float[]     | [-1, -1.5] as float[]          || [ 2.0, 3.0, -1.0, -1.5 ] as float[]

    }

}
