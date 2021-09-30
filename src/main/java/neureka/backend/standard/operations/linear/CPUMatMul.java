package neureka.backend.standard.operations.linear;

class CPUMatMul {

    /*
    static double[] A = { 4.00, 3.00,
                          2.00, 1.00 };

    static double[] B = { -0.500, 1.500,
                          1.000, -2.0000 };

    public static void main(String[] args) {
        System.out.println(Arrays.toString(multiplicar(
                A, new int[]{2, 2}, B, new int[]{2, 2}
        )));
    }
    */

    public static double[] multiplicar(
            double[] A, int[] shapeA, double[] B, int[] shapeB
    ) {
        // A * B = C
        int aRows = shapeA[0];
        int aColumns = shapeA[1];
        int bRows = shapeB[0];
        int bColumns = shapeB[1];

        if (aColumns != bRows) {
            throw new IllegalArgumentException("A:Rows: " + aColumns + " did not match B:Columns " + bRows + ".");
        }

        double[] C = new double[aRows*bColumns];
        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bColumns; j++) {
                C[i*bColumns+j] = 0.00000;
            }
        }

        for (int i = 0; i < aRows; i++) { // aRow
            for (int j = 0; j < bColumns; j++) { // bColumn
                for (int k = 0; k < aColumns; k++) { // aColumn
                    C[i*bColumns+j] += A[i*aColumns+k] * B[k*bColumns+j];
                }
            }
        }
        return C;
    }

}