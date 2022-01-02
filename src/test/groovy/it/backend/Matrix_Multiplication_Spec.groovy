package it.backend

import neureka.Tsr
import spock.lang.Specification

class Matrix_Multiplication_Spec extends Specification {

    def 'The simple CPU matrix multiplication implementation works as expected.'(
            Class<Object> type, int M, int K, int N, Object A, Object B, def expectedC
    ) {
        given :
            Tsr<?> a = Tsr.of(type, [M,K] as int[], A)
            Tsr<?> b = Tsr.of(type, [K,N] as int[], B)

        when :
            Tsr<?> c = a.matMul(b)

        then :
            c.shape() == [M,N]
        and :
            c.value == expectedC

        where :
            type  | M | K | N | A                     | B                              || expectedC
           Double | 2 | 2 | 2 | [4,3,2,1] as double[] | [-0.5, 1.5, 1, -2] as double[] || [ 1, 0, 0, 1 ] as double[]
           Double | 1 | 2 | 1 | [-2,1] as double[]    | [-1, -1.5] as double[]         || [ 0.5 ] as double[]
           Double | 2 | 1 | 2 | [-2,1] as double[]    | [-1, -1.5] as double[]         || [ 2.0, 3.0, -1.0, -1.5 ] as double[]
           Float  | 2 | 2 | 2 | [4,3,2,1] as float[]  | [-0.5, 1.5, 1, -2] as float[]  || [ 1, 0, 0, 1 ] as float[]
           Float  | 1 | 2 | 1 | [-2,1] as float[]     | [-1, -1.5] as float[]          || [ 0.5 ] as float[]
           Float  | 2 | 1 | 2 | [-2,1] as float[]     | [-1, -1.5] as float[]          || [ 2.0, 3.0, -1.0, -1.5 ] as float[]

    }

}
