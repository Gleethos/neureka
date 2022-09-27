package ut.backend.core

import neureka.backend.main.operations.linear.CPUMatMul
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Internal CPU based Matrix Multiplication")
@Narrative('''

    This specification covers library internal matrix multiplication logic,
    specifically the CPU implementation.
    Do not depend on the API used in this specification as it is subject to change!

''')
class Matrix_Multiplication_Spec extends Specification
{
    def 'The CPU matrix multiplication implementation works as expected.'(
            def type, def A, def B, int M, int K, int N, def expectedC
    ) {
        given : 'We crete an output array and convert it to the targeted array type.'
            var C = new double[M*N].asType(type)

        when : 'We perform the matrix multiplication.'
            CPUMatMul.execute(true, A.asType(type), B.asType(type), C, M, K, N)

        then : 'The result is as expected.'
            C == expectedC

        where : 'The following data arrays and dimensions can be used for the test.'
            type     | A            | B                  | M | K | N || expectedC
            float[]  | [4, 3, 2, 1] | [-0.5, 1.5, 1, -2] | 2 | 2 | 2 || [ 1, 0, 0, 1 ]
            float[]  | [-2, 1]      | [-1, -1.5]         | 1 | 2 | 1 || [ 0.5 ]
            float[]  | [-2, 1]      | [-1, -1.5]         | 2 | 1 | 2 || [ 2.0, 3.0, -1.0, -1.5 ]
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
        expect: 'The test script runs without errors.'
            new InternalMatMulTest().start()
    }

}
