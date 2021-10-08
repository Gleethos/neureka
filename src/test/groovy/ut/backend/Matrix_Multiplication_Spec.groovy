package ut.backend

import neureka.backend.standard.operations.linear.SimpleMatMul;
import spock.lang.Specification;

class Matrix_Multiplication_Spec extends Specification {

    def 'The simple CPU matrix multiplication implementation works as expected.'(
            double[] A, double[] B, int M, int K, int N, double[] expectedC
    ) {
        given :
            def C = new double[M*N]

        when :
            SimpleMatMul.execute(A, B, C, M, K, N)

        then :
            C == expectedC

        where :
            A            | B                  | M | K | N || expectedC
            [4, 3, 2, 1] | [-0.5, 1.5, 1, -2] | 2 | 2 | 2 || [ 1, 0, 0, 1 ]
            [-2, 1]      | [-1, -1.5]         | 1 | 2 | 1 || [ 0.5 ]
            [-2, 1]      | [-1, -1.5]         | 2 | 1 | 2 || [ 2.0, 3.0, -1.0, -1.5 ]

    }

}