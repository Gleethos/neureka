package neureka.backend.standard.operations.linear;

import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.standard.operations.linear.internal.Conf;
import neureka.backend.standard.operations.linear.internal.M32;
import neureka.backend.standard.operations.linear.internal.M64;
import neureka.devices.host.CPU;
import neureka.ndim.AbstractNDArray;
import neureka.ndim.config.NDConfiguration;

public class SimpleMatMul implements ImplementationFor<CPU> {

    public static void execute(
            double[] A, double[] B, double[] C, int aRows, int aCols, int bCols
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
        M64 a = new M64(aRows, aCols, A);
        M64 b = new M64(aCols, bCols, B);
        a.multiply(b, C);
    }

    public static void execute(
            float[] A, float[] B, float[] C, int aRows, int aCols, int bCols
    ) {
        M32 a = new M32(aRows, aCols, A);
        M32 b = new M32(aCols, bCols, B);
        a.multiply(b, C);
    }

    @Override
    public void run( ExecutionCall<CPU> call ) {

        if ( !call.validate().allShare( t -> t.getNDConf().getLayout() ).isValid() )
            throw new IllegalArgumentException(
                    "Data layout inconsistency between provided tensors encountered. " +
                    "All tensors must be of the same layout."
            );

        if ( !call.validate().allShare(AbstractNDArray::getDataType).isValid() )
            throw new IllegalArgumentException(
               "Type inconsistency between provided tensors encountered. " +
               "All tensors must be of the same type."
            );

        NDConfiguration.Layout layout = call.tensor( 1 ).getNDConf().getLayout();

        Conf.ROW_MAJOR = ( layout == NDConfiguration.Layout.ROW_MAJOR );

        int[] shapeA = call.tensor( 1 ).getNDConf().shape();
        int[] shapeB = call.getTensors()[ 2 ].getNDConf().shape();
        int[] shapeC = call.tensor( 0 ).getNDConf().shape();

        // A * B = C // [MxK]*[KxN] = [MxN]
        int aRows = shapeA[0];
        int aCols = shapeA[1];
        int bRows = shapeB[0];
        int bCols = shapeB[1];

        if ( aCols != bRows )
            throw new IllegalArgumentException("A:Rows: " + aCols + " did not match B:Columns " + bRows + ".");

        Class<?> type = call.tensor( 0 ).getDataType().getJVMTypeClass();
        if ( type == Double.class ) {
            double[] A = (double[]) call.getTsrOfType(Double.class, 1).getData();
            double[] B = (double[]) call.getTsrOfType(Double.class, 2).getData();
            double[] C = (double[]) call.getTsrOfType(Double.class, 0).getData();

            execute(A, B, C, aRows, aCols, bCols);
        } else if ( type == Float.class ) {
            float[] A = (float[]) call.getTsrOfType(Float.class, 1).getData();
            float[] B = (float[]) call.getTsrOfType(Float.class, 2).getData();
            float[] C = (float[]) call.getTsrOfType(Float.class, 0).getData();

            execute(A, B, C, aRows, aCols, bCols);
        }
    }
}