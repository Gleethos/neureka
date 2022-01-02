package neureka.backend.standard.operations.linear;

import neureka.backend.standard.operations.linear.internal.M64;

public class SimpleMatMul {

    public static void execute(
            double[] A,
            int[] shapeA,
            double[] B,
            int[] shapeB,
            double[] C,
            int[] shapeC
    ) {
        // A * B = C // [MxK]*[KxN] = [MxN]
        int aRows = shapeA[0];
        int aCols = shapeA[1];
        int bRows = shapeB[0];
        int bCols = shapeB[1];

        if ( aCols != bRows ) {
            throw new IllegalArgumentException("A:Rows: " + aCols + " did not match B:Columns " + bRows + ".");
        }

        execute(A, B, C, aRows, aCols, bCols);
    }

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

}