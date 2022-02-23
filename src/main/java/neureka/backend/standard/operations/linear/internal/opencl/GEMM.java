package neureka.backend.standard.operations.linear.internal.opencl;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.opencl.KernelCaller;
import neureka.devices.opencl.OpenCLDevice;

import java.util.function.Supplier;

public class GEMM implements ImplementationFor<OpenCLDevice> {

    @Override
    public void run( ExecutionCall<OpenCLDevice> call) {

            Tsr<Float> c = call.getTsrOfType(Float.class, 0);
            Tsr<Float> a = call.getTsrOfType(Float.class, 1);
            Tsr<Float> b = call.getTsrOfType(Float.class, 2);

            assert a.shape(1) == b.shape(0);

            int M = a.shape(0);
            int K = a.shape(1);
            int N = b.shape(1);

            String kernelName = "fast_CM_MM_"+M+"x"+K+"x"+N+"";

            // Determining optimal tile widths
            int MW = 1;
            int KW = 1;

            for ( int s : new int[]{16,8,4,2,1} )
            if ( M % s == 0 ) { MW = s; break; }
            for ( int s : new int[]{8,4,2,1} )
            if ( N % s == 0 && K % s == 0 ) { KW = s; break; }

            int NW = KW;

        int finalMW = MW;
        int finalKW = KW;

        Supplier<String> code = () ->
                    "   #define K "+K+"                                                                                 \n" +
                    "   #define N "+N+"                                                                                 \n" +
                    "   #define MW "+ finalMW +"     // M tile Width                                                    \n" +
                    "   #define NW "+NW+"     // N tile Width  -- NW & KW should be the same !                          \n" +
                    "   #define KW "+ finalKW +"     // K tile Width                                                    \n" +
                    "   #define MT "+(int)Math.floor(M/ finalMW)+"   // MT is max for 'mt' (M tile count)               \n" +
                    "   #define KT "+(int)Math.floor(K/ finalKW)+"   // KT is max for 'kt' (K tile count)               \n" +
                    "   #define floatMW "+(finalMW != 1 ? "float"+ finalMW : "float")+"                                 \n" +
                    "   #define floatKW "+(finalKW != 1 ? "float"+ finalKW : "float")+"                                 \n" +
                    "   __kernel void "+kernelName+"(                                                                   \n" +
                    "               const __global floatMW* restrict A,                                                 \n" +
                    "               const __global floatKW* restrict B,                                                 \n" +
                    "                     __global floatMW* C                                                           \n" +
                    "   ) {{                                                                                            \n" +
                    "       size_t mt    = get_global_id(0);    //global M-tile id                                      \n" +
                    "       size_t nc    = get_global_id(1);    //global N-tile id                                      \n" +
                    "       size_t batch = get_global_id(2);                                                            \n" +
                    "                                                                                                   \n" +
                    "       float AT[KW][MW]; // sub tiles                                                              \n" +
                    "       float BT[NW][KW];                                                                           \n" +
                    "       float CT[NW][MW];                                                                           \n" +
                    "       #pragma unroll                                                                              \n" +
                    "       for ( uint i=0; i<NW*MW; ++i ) // zero CT tile                                              \n" +
                    "           ((float*) CT)[i] = 0.0;                                                                 \n" +
                    "       for ( uint kt=0; kt<KT; ++kt )  // iterate over K-dim tiles                                 \n" +
                    "       {{                                                                                          \n" +
                    "           #pragma unroll                                                                          \n" +
                    "           for ( uint k=0; k<KW; ++k )  // every k-element inside K-dim tile                       \n" +
                    "               *( (floatMW*) AT[k] ) = A[batch*K*MT + (kt*KW + k)*MT + mt]; // store M-Width floats\n" +
                    "           #pragma unroll                                                                          \n" +
                    "           for ( uint n=0; n<NW; ++n )  // every n-element inside N-dim tile                       \n" +
                    "               *( (floatKW*) BT[n] ) = B[batch*N*KT + (nc*NW + n)*KT + kt]; // store K-Width floats\n" +
                    "           #pragma unroll                                                                          \n" +
                    "           for ( uint k=0; k<KW; ++k )                                                             \n" +
                    "           #pragma unroll                                                                          \n" +
                    "           for ( uint n=0; n<NW; ++n )  // sub tiles multiplication                                \n" +
                    "           #pragma unroll                                                                          \n" +
                    "           for ( uint m=0; m<MW; ++m )                                                             \n" +
                    "               CT[n][m] += AT[k][m] * BT[n][k];                                                    \n" +
                    "       }}                                                                                          \n" +
                    "       #pragma unroll                                                                              \n" +
                    "       for ( uint n = 0; n < NW; ++n )                                                             \n" +
                    "           C[ batch * N * MT + ( nc * NW + n ) * MT + mt ] += *( (floatMW*) CT[n] );               \n" +
                    "   }}                                                                                                ";

            //return new KernelCode(kernelName, code);

        KernelCaller caller =
             call.getDevice().hasAdHocKernel(kernelName)
                 ? call.getDevice().getAdHocKernel(kernelName)
                 : call.getDevice().compileAdHocKernel(kernelName, code.get()).getAdHocKernel(kernelName);


        long[] local =  null; // This kernel does not have local memory (uses register/private memory instead)
        long[] global = new long[]{(long) Math.floor(M/MW), (long) Math.floor(N/NW), 1 };


        caller.pass( a ).pass( b ).pass( c ).call( global, local );
    }

}
