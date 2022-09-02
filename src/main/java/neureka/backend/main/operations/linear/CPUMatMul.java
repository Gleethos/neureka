package neureka.backend.main.operations.linear;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.operations.linear.internal.blas.GEMM;
import neureka.devices.host.CPU;
import neureka.ndim.config.NDConfiguration;

/**
 *  This is a library internal class, do not depend on this.
 */
public class CPUMatMul implements ImplementationFor<CPU> {

    public static void execute(
            boolean rowMajor, double[] A, double[] B, double[] C, int aRows, int aCols, int bCols
    ) {
        /* // Use this for verifying validity:
            for ( int i = 0; i < aRows; i++ ) { // aRow
                for ( int j = 0; j < bCols; j++ ) { // bColumn
                    for ( int k = 0; k < aCols; k++ ) { // aColumn
                        C[ i * bCols + j ] += A[ i * aCols + k ] * B[ k * bCols + j ];
                    }
                }
            }
        */
        GEMM.operationForF64( rowMajor, aRows, bCols ).invoke( C, A, aCols, B );
    }

    public static void execute(
            boolean rowMajor, float[] A, float[] B, float[] C, int aRows, int aCols, int bCols
    ) {
        GEMM.operationForF32( rowMajor, aRows, bCols ).invoke( C, A, aCols, B );
    }

    @Override
    public Tsr<?> run(ExecutionCall<CPU> call )
    {
        if ( !call.validate().all( (t1, t2) -> t1.getNDConf().getLayout().isCompatible(t2.getNDConf().getLayout()) ).isValid() )
            throw new IllegalArgumentException(
                        "Data layout inconsistency between provided tensors encountered. " +
                        "All tensors must be of the same layout."
                    );

        if ( !call.validate().allShare(Tsr::getDataType).isValid() )
            throw new IllegalArgumentException(
                       "Type inconsistency between provided tensors encountered. " +
                       "All tensors must be of the same type."
                    );

        NDConfiguration.Layout layout = call.input( 1 ).getNDConf().getLayout();

        boolean rowMajor = ( layout == NDConfiguration.Layout.ROW_MAJOR );

        int[] shapeA = call.input( 1 ).getNDConf().shape();
        int[] shapeB = call.input( 2 ).getNDConf().shape();
        int[] shapeC = call.input( 0 ).getNDConf().shape();

        // A * B = C // [MxK]*[KxN] = [MxN]
        int aRows = shapeA[0];
        int aCols = shapeA[1];
        int bRows = shapeB[0];
        int bCols = shapeB[1];

        if ( aCols != bRows )
            throw new IllegalArgumentException("'A' matrix rows " + aCols + " did not match 'B' matrix columns " + bRows + ".");

        Class<?> type = call.input( 0 ).getDataType().getItemTypeClass();
        if ( type == Double.class ) {
            double[] A = call.input(Double.class, 1).getUnsafe().getDataAs(double[].class);
            double[] B = call.input(Double.class, 2).getUnsafe().getDataAs(double[].class);
            double[] C = call.input(Double.class, 0).getUnsafe().getDataForWriting(double[].class);

            execute( rowMajor, A, B, C, aRows, aCols, bCols );
        } else if ( type == Float.class ) {
            float[] A = call.input(Float.class, 1).getUnsafe().getDataAs(float[].class);
            float[] B = call.input(Float.class, 2).getUnsafe().getDataAs(float[].class);
            float[] C = call.input(Float.class, 0).getUnsafe().getDataForWriting(float[].class);

            execute( rowMajor, A, B, C, aRows, aCols, bCols );
        }
        else
            throw new IllegalArgumentException(
                        "Data type '"+type.getSimpleName()+"' not yet supported " +
                        "for CPU based matrix multiplication!"
                    );

        return call.input( 0 );
    }
}