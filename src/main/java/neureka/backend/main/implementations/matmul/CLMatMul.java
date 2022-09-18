package neureka.backend.main.implementations.matmul;

import neureka.backend.main.implementations.SimpleCLImplementation;
import neureka.backend.main.operations.linear.internal.opencl.CLGEMM;
import neureka.ndim.config.NDConfiguration;

public class CLMatMul extends SimpleCLImplementation
{
    public CLMatMul() {
        super(
            call -> {
                if (
                        call.validate()
                                .all( t -> t.getNDConf().getLayout() == NDConfiguration.Layout.COLUMN_MAJOR )
                                .isValid()
                ) {
                    return new CLGEMM().run( call );
                } else {
                    int M = call.input(1).shape(0);
                    int N = call.input(2).shape(1);
                    int K = call.input(1).shape(1);
                    call.getDevice()
                            .getKernel(call)
                            .pass(M).pass(N).pass(K)
                            .pass(call.input(Number.class, 1))
                            .pass(call.input(Number.class, 2))
                            .pass(call.input(Number.class, 0))
                            .call(new long[]{M, N}, null);

                    return call.input(0);
                }
            },
            3,
            "simple_matMul",
            "   __kernel void simple_matMul(                                         \n" +
            "          const int M, const int N, const int K,                        \n" +
            "          const __global float* A,                                      \n" +
            "          const __global float* B,                                      \n" +
            "                __global float* C                                       \n" +
            "   ) {                                                                  \n" +
            "       const int m = get_global_id(0); // Row index of C (0..M)         \n" +
            "       const int n = get_global_id(1); // Col index of C (0..N)         \n" +
            "                                                                        \n" +
            "       // Compute a single element (loop over K)                        \n" +
            "       float acc = 0.0f;                                                \n" +
            "       for ( int k = 0; k < K; k++ )                                    \n" +
            "           acc += A[ k + m * K ] * B[ n + k * N ];                      \n" +
            "                                                                        \n" +
            "       // Store the result                                              \n" +
            "       C[ n + m * N ] = acc;                                            \n" +
            "   }                                                                    \n"
        );
    }
}
