package ut.backend.core


import neureka.backend.main.operations.linear.CPUMatMul
import spock.lang.Specification
import ut.backend.core.InternalMatMulTest

class Matrix_Multiplication_Spec extends Specification
{

    def 'The CPU matrix multiplication implementation works as expected.'(
            def type, def A, def B, int M, int K, int N, def expectedC
    ) {
        given :
            var C = new double[M*N].asType(type)

        when :
            CPUMatMul.execute(true, A.asType(type), B.asType(type), C, M, K, N)

        then :
            C == expectedC

        where :
            type     | A            | B                  | M | K | N || expectedC
            double[] | [4, 3, 2, 1] | [-0.5, 1.5, 1, -2] | 2 | 2 | 2 || [ 1, 0, 0, 1 ]
            double[] | [-2, 1]      | [-1, -1.5]         | 1 | 2 | 1 || [ 0.5 ]
            double[] | [-2, 1]      | [-1, -1.5]         | 2 | 1 | 2 || [ 2.0, 3.0, -1.0, -1.5 ]
            int[]    | [4, 3, 2, 1] | [-0.5, 1.5, 1, -2] | 2 | 2 | 2 || [ 3, -2, 1, 0 ]
            int[]    | [-2, 1]      | [-1, -1.5]         | 1 | 2 | 1 || [ 1 ]
            int[]    | [-2, 1]      | [-1, -1.5]         | 2 | 1 | 2 || [ 2, 2, -1, -1 ]
            long[]   | [4, 3, 2, 1] | [-0.5, 1.5, 1, -2] | 2 | 2 | 2 || [ 3, -2, 1, 0 ]
            long[]   | [-2, 1]      | [-1, -1.5]         | 1 | 2 | 1 || [ 1 ]
            long[]   | [-2, 1]      | [-1, -1.5]         | 2 | 1 | 2 || [ 2, 2, -1, -1 ]
    }

    def 'The internal matrix multiplication test script runs!'()
    {
        expect:
            new InternalMatMulTest().start()
    }

}
