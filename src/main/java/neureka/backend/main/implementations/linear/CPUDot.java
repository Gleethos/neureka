package neureka.backend.main.implementations.linear;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.operations.linear.internal.blas.DOT;
import neureka.devices.host.CPU;

public class CPUDot implements ImplementationFor<CPU> {

    @Override
    public Tensor<?> run(ExecutionCall<CPU> call) {

        if ( !call.validate().all( (t1, t2) -> t1.getNDConf().getLayout().isCompatible(t2.getNDConf().getLayout()) ).isValid() )
            throw new IllegalArgumentException(
                        "Data layout inconsistency between provided tensors encountered. " +
                        "All tensors must be of the same layout."
                    );

        if ( !call.validate().allShare(Tensor::getDataType).isValid() )
            throw new IllegalArgumentException(
                       "Type inconsistency between provided tensors encountered. " +
                       "All tensors must be of the same type."
                    );

        int[] shapeA = call.input( 1 ).getNDConf().shape();
        int[] shapeB = call.input( 2 ).getNDConf().shape();
        int[] shapeC = call.input( 0 ).getNDConf().shape();

        if ( shapeA.length != 1 || shapeB.length != 1 || shapeC.length != 1 )
            throw new IllegalArgumentException("Dot product only works on vectors.");

        if ( shapeA[0] != shapeB[0] )
            throw new IllegalArgumentException("Dot product only works on vectors of the same length.");

        // A * B = C // [N]*[N] = [1]
        int size = shapeA[0];

        Class<?> type = call.input( 0 ).getDataType().getItemTypeClass();
        if ( type == Double.class ) {
            double[] A = call.input(Double.class, 1).mut().getDataAs(double[].class);
            double[] B = call.input(Double.class, 2).mut().getDataAs(double[].class);
            double[] C = call.input(Double.class, 0).mut().getDataForWriting(double[].class);
            execute( A, B, C, size );
        } else if ( type == Float.class ) {
            float[] A = call.input(Float.class, 1).mut().getDataAs(float[].class);
            float[] B = call.input(Float.class, 2).mut().getDataAs(float[].class);
            float[] C = call.input(Float.class, 0).mut().getDataForWriting(float[].class);
            execute( A, B, C, size );
        }
        else if ( type == Long.class ) {
            long[] A = call.input(Long.class, 1).mut().getDataAs(long[].class);
            long[] B = call.input(Long.class, 2).mut().getDataAs(long[].class);
            long[] C = call.input(Long.class, 0).mut().getDataForWriting(long[].class);
            execute( A, B, C, size );
        }
        else if ( type == Integer.class ) {
            int[] A = call.input(Integer.class, 1).mut().getDataAs(int[].class);
            int[] B = call.input(Integer.class, 2).mut().getDataAs(int[].class);
            int[] C = call.input(Integer.class, 0).mut().getDataForWriting(int[].class);
            execute( A, B, C, size );
        }
        else
            throw new IllegalArgumentException(
                        "Data type '"+type.getSimpleName()+"' not yet supported " +
                        "for CPU based dot product!"
                    );

        return call.input( 0 );
    }

    private static void execute( double[] A, double[] B, double[] C, int size ) {
        C[0] = DOT.invoke( A, 0, B, 0, 0, size );
    }

    private static void execute( float[] A, float[] B, float[] C, int size ) {
        C[0] = DOT.invoke( A, 0, B, 0, 0, size );
    }

    private static void execute( long[] A, long[] B, long[] C, int size ) {
        C[0] = DOT.invoke( A, 0, B, 0, 0, size );
    }

    private static void execute( int[] A, int[] B, int[] C, int size ) {
        C[0] = DOT.invoke( A, 0, B, 0, 0, size );
    }

}
