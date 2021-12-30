/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.operation;

import neureka.backend.standard.operations.linear.fast.array.operation.AXPY;
import neureka.backend.standard.operations.linear.fast.array.operation.DOT;
import neureka.backend.standard.operations.linear.fast.concurrent.Parallelism;
import neureka.backend.standard.operations.linear.fast.concurrent.ProcessingService;
import neureka.backend.standard.operations.linear.fast.concurrent.WorkScheduler;
import neureka.backend.standard.operations.linear.fast.matrix.store.MatrixCore;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.structure.Structure2D;

import java.util.Arrays;
import java.util.function.IntSupplier;

public class MultiplyRight implements MatrixOperation {

    @FunctionalInterface
    public interface Primitive32 {
        void invoke(
                float[] product,
                float[] left,
                int complexity,
                Access1D<?> right
        );
    }

    @FunctionalInterface
    public interface Primitive64 {
        void invoke(
                double[] product,
                double[] left,
                int complexity,
                Access1D<?> right
        );
    }

    public static IntSupplier PARALLELISM = Parallelism.THREADS;
    public static int THRESHOLD = 32;

    private static final WorkScheduler.Divider DIVIDER = ProcessingService.INSTANCE.divider();

    public static Primitive32 newPrimitive32(final long rows, final long columns) {
        if (rows > THRESHOLD && columns > THRESHOLD) {
            return MultiplyRight::fillMxN_MT;
        }
        if (columns == 1) {
            return MultiplyRight::fillMx1;
        }
        if (rows == 1) {
            return MultiplyRight::fill1xN;
        }
        return MultiplyRight::fillMxN;
    }

    public static Primitive64 newPrimitive64(final long rows, final long columns) {
        if (rows > THRESHOLD && columns > THRESHOLD) {
            return MultiplyRight::fillMxN_MT;
        }
        if (rows == 5 && columns == 5) return MultiplyRight::fill5x5;
        if (rows == 4 && columns == 4) return MultiplyRight::fill4x4;
        if (rows == 3 && columns == 3) return MultiplyRight::fill3x3;
        if (rows == 2 && columns == 2) return MultiplyRight::fill2x2;
        if (rows == 1 && columns == 1) return MultiplyRight::fill1x1;
        if (columns == 1) return MultiplyRight::fillMx1;
        if (rows == 10) return MultiplyRight::fill0xN;
        if (rows == 9) return MultiplyRight::fill9xN;
        if (rows == 8) return MultiplyRight::fill8xN;
        if (rows == 7) return MultiplyRight::fill7xN;
        if (rows == 6) return MultiplyRight::fill6xN;
        if (rows == 1) return MultiplyRight::fill1xN;
        return MultiplyRight::fillMxN;
    }

    static void addMx1(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        int nbRows = product.length;
        for (int c = 0; c < complexity; c++) {
            AXPY.invoke(product, 0, right.doubleValue(c), left, c * nbRows, 0, nbRows);
        }
    }

    static void addMx1(final float[] product, final float[] left, final int complexity, final Access1D<?> right) {

        int nbRows = product.length;
        for (int c = 0; c < complexity; c++) {
            AXPY.invoke(product, 0, right.floatValue(c), left, c * nbRows, 0, nbRows);
        }
    }

    static void addMxC(final double[] product, final int firstColumn, final int columnLimit, final double[] left, final int complexity,
            final Access1D<?> right) {

        int nbRows = left.length / complexity;

        for (int c = 0; c < complexity; c++) {

            int firstInRightRow = MatrixCore.firstInRow(right, c, firstColumn);
            int limitOfRightRow = MatrixCore.limitOfRow(right, c, columnLimit);

            for (int j = firstInRightRow; j < limitOfRightRow; j++) {
                AXPY.invoke(product, j * nbRows, right.doubleValue(Structure2D.index(complexity, c, j)), left, c * nbRows, 0, nbRows);
            }
        }
    }

    static void addMxC(final float[] product, final int firstColumn, final int columnLimit, final float[] left, final int complexity, final Access1D<?> right) {

        int nbRows = left.length / complexity;

        for (int c = 0; c < complexity; c++) {

            int firstInRightRow = MatrixCore.firstInRow(right, c, firstColumn);
            int limitOfRightRow = MatrixCore.limitOfRow(right, c, columnLimit);

            for (int j = firstInRightRow; j < limitOfRightRow; j++) {
                AXPY.invoke(product, j * nbRows, right.floatValue(Structure2D.index(complexity, c, j)), left, c * nbRows, 0, nbRows);
            }
        }
    }

    static void addMxN_MT(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {
        MultiplyRight.divide(0, Math.toIntExact(right.size() / complexity), (f, l) -> MultiplyRight.addMxC(product, f, l, left, complexity, right));
    }

    static void addMxN_MT(final float[] product, final float[] left, final int complexity, final Access1D<?> right) {
        MultiplyRight.divide(0, Math.toIntExact(right.size() / complexity), (f, l) -> MultiplyRight.addMxC(product, f, l, left, complexity, right));
    }

    static void divide(final int first, final int limit, final WorkScheduler.Worker worker) {
        DIVIDER.parallelism(PARALLELISM).threshold(THRESHOLD).divide(first, limit, worker);
    }

    static void fill0xN(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        int tmpRowDim = 10;
        int tmpColDim = product.length / tmpRowDim;

        for (int j = 0; j < tmpColDim; j++) {

            double tmp0J = 0.0;
            double tmp1J = 0.0;
            double tmp2J = 0.0;
            double tmp3J = 0.0;
            double tmp4J = 0.0;
            double tmp5J = 0.0;
            double tmp6J = 0.0;
            double tmp7J = 0.0;
            double tmp8J = 0.0;
            double tmp9J = 0.0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                double tmpRightCJ = right.doubleValue(Structure2D.index(complexity, c, j));
                tmp0J += left[tmpIndex++] * tmpRightCJ;
                tmp1J += left[tmpIndex++] * tmpRightCJ;
                tmp2J += left[tmpIndex++] * tmpRightCJ;
                tmp3J += left[tmpIndex++] * tmpRightCJ;
                tmp4J += left[tmpIndex++] * tmpRightCJ;
                tmp5J += left[tmpIndex++] * tmpRightCJ;
                tmp6J += left[tmpIndex++] * tmpRightCJ;
                tmp7J += left[tmpIndex++] * tmpRightCJ;
                tmp8J += left[tmpIndex++] * tmpRightCJ;
                tmp9J += left[tmpIndex++] * tmpRightCJ;
            }

            product[tmpIndex = j * tmpRowDim] = tmp0J;
            product[++tmpIndex] = tmp1J;
            product[++tmpIndex] = tmp2J;
            product[++tmpIndex] = tmp3J;
            product[++tmpIndex] = tmp4J;
            product[++tmpIndex] = tmp5J;
            product[++tmpIndex] = tmp6J;
            product[++tmpIndex] = tmp7J;
            product[++tmpIndex] = tmp8J;
            product[++tmpIndex] = tmp9J;
        }
    }

    static void fill1x1(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        double tmp00 = 0.0;

        int tmpLeftStruct = left.length / complexity; // The number of rows in the product- and left-matrix.

        for (int c = 0; c < complexity; c++) {
            tmp00 += left[c * tmpLeftStruct] * right.doubleValue(c);
        }

        product[0] = tmp00;
    }

    static void fill1xN(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        for (int j = 0, nbCols = product.length; j < nbCols; j++) {
            int firstInCol = MatrixCore.firstInColumn(right, j, 0);
            int limitOfCol = MatrixCore.firstInColumn(right, j, complexity);
            product[j] = DOT.invoke(left, 0, right, j * complexity, firstInCol, limitOfCol);
        }
    }

    static void fill1xN(final float[] product, final float[] left, final int complexity, final Access1D<?> right) {

        for (int j = 0, nbCols = product.length; j < nbCols; j++) {
            int firstInCol = MatrixCore.firstInColumn(right, j, 0);
            int limitOfCol = MatrixCore.firstInColumn(right, j, complexity);
            product[j] = DOT.invoke(left, 0, right, j * complexity, firstInCol, limitOfCol);
        }
    }

    static void fill2x2(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        double tmp00 = 0.0;
        double tmp10 = 0.0;
        double tmp01 = 0.0;
        double tmp11 = 0.0;

        int tmpIndex;
        for (int c = 0; c < complexity; c++) {

            tmpIndex = c * 2;
            double tmpLeft0 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft1 = left[tmpIndex];
            tmpIndex = c;
            double tmpRight0 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight1 = right.doubleValue(tmpIndex);

            tmp00 += tmpLeft0 * tmpRight0;
            tmp10 += tmpLeft1 * tmpRight0;
            tmp01 += tmpLeft0 * tmpRight1;
            tmp11 += tmpLeft1 * tmpRight1;
        }

        product[0] = tmp00;
        product[1] = tmp10;
        product[2] = tmp01;
        product[3] = tmp11;
    }

    static void fill3x3(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        double tmp00 = 0.0;
        double tmp10 = 0.0;
        double tmp20 = 0.0;
        double tmp01 = 0.0;
        double tmp11 = 0.0;
        double tmp21 = 0.0;
        double tmp02 = 0.0;
        double tmp12 = 0.0;
        double tmp22 = 0.0;

        int tmpIndex;
        for (int c = 0; c < complexity; c++) {

            tmpIndex = c * 3;
            double tmpLeft0 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft1 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft2 = left[tmpIndex];
            tmpIndex = c;
            double tmpRight0 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight1 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight2 = right.doubleValue(tmpIndex);

            tmp00 += tmpLeft0 * tmpRight0;
            tmp10 += tmpLeft1 * tmpRight0;
            tmp20 += tmpLeft2 * tmpRight0;
            tmp01 += tmpLeft0 * tmpRight1;
            tmp11 += tmpLeft1 * tmpRight1;
            tmp21 += tmpLeft2 * tmpRight1;
            tmp02 += tmpLeft0 * tmpRight2;
            tmp12 += tmpLeft1 * tmpRight2;
            tmp22 += tmpLeft2 * tmpRight2;
        }

        product[0] = tmp00;
        product[1] = tmp10;
        product[2] = tmp20;
        product[3] = tmp01;
        product[4] = tmp11;
        product[5] = tmp21;
        product[6] = tmp02;
        product[7] = tmp12;
        product[8] = tmp22;
    }

    static void fill4x4(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        double tmp00 = 0.0;
        double tmp10 = 0.0;
        double tmp20 = 0.0;
        double tmp30 = 0.0;
        double tmp01 = 0.0;
        double tmp11 = 0.0;
        double tmp21 = 0.0;
        double tmp31 = 0.0;
        double tmp02 = 0.0;
        double tmp12 = 0.0;
        double tmp22 = 0.0;
        double tmp32 = 0.0;
        double tmp03 = 0.0;
        double tmp13 = 0.0;
        double tmp23 = 0.0;
        double tmp33 = 0.0;

        int tmpIndex;
        for (int c = 0; c < complexity; c++) {

            tmpIndex = c * 4;
            double tmpLeft0 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft1 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft2 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft3 = left[tmpIndex];
            tmpIndex = c;
            double tmpRight0 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight1 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight2 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight3 = right.doubleValue(tmpIndex);

            tmp00 += tmpLeft0 * tmpRight0;
            tmp10 += tmpLeft1 * tmpRight0;
            tmp20 += tmpLeft2 * tmpRight0;
            tmp30 += tmpLeft3 * tmpRight0;
            tmp01 += tmpLeft0 * tmpRight1;
            tmp11 += tmpLeft1 * tmpRight1;
            tmp21 += tmpLeft2 * tmpRight1;
            tmp31 += tmpLeft3 * tmpRight1;
            tmp02 += tmpLeft0 * tmpRight2;
            tmp12 += tmpLeft1 * tmpRight2;
            tmp22 += tmpLeft2 * tmpRight2;
            tmp32 += tmpLeft3 * tmpRight2;
            tmp03 += tmpLeft0 * tmpRight3;
            tmp13 += tmpLeft1 * tmpRight3;
            tmp23 += tmpLeft2 * tmpRight3;
            tmp33 += tmpLeft3 * tmpRight3;
        }

        product[0] = tmp00;
        product[1] = tmp10;
        product[2] = tmp20;
        product[3] = tmp30;
        product[4] = tmp01;
        product[5] = tmp11;
        product[6] = tmp21;
        product[7] = tmp31;
        product[8] = tmp02;
        product[9] = tmp12;
        product[10] = tmp22;
        product[11] = tmp32;
        product[12] = tmp03;
        product[13] = tmp13;
        product[14] = tmp23;
        product[15] = tmp33;
    }

    static void fill5x5(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        double tmp00 = 0.0;
        double tmp10 = 0.0;
        double tmp20 = 0.0;
        double tmp30 = 0.0;
        double tmp40 = 0.0;
        double tmp01 = 0.0;
        double tmp11 = 0.0;
        double tmp21 = 0.0;
        double tmp31 = 0.0;
        double tmp41 = 0.0;
        double tmp02 = 0.0;
        double tmp12 = 0.0;
        double tmp22 = 0.0;
        double tmp32 = 0.0;
        double tmp42 = 0.0;
        double tmp03 = 0.0;
        double tmp13 = 0.0;
        double tmp23 = 0.0;
        double tmp33 = 0.0;
        double tmp43 = 0.0;
        double tmp04 = 0.0;
        double tmp14 = 0.0;
        double tmp24 = 0.0;
        double tmp34 = 0.0;
        double tmp44 = 0.0;

        int tmpIndex;
        for (int c = 0; c < complexity; c++) {

            tmpIndex = c * 5;
            double tmpLeft0 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft1 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft2 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft3 = left[tmpIndex];
            tmpIndex++;
            double tmpLeft4 = left[tmpIndex];
            tmpIndex = c;
            double tmpRight0 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight1 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight2 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight3 = right.doubleValue(tmpIndex);
            tmpIndex += complexity;
            double tmpRight4 = right.doubleValue(tmpIndex);

            tmp00 += tmpLeft0 * tmpRight0;
            tmp10 += tmpLeft1 * tmpRight0;
            tmp20 += tmpLeft2 * tmpRight0;
            tmp30 += tmpLeft3 * tmpRight0;
            tmp40 += tmpLeft4 * tmpRight0;
            tmp01 += tmpLeft0 * tmpRight1;
            tmp11 += tmpLeft1 * tmpRight1;
            tmp21 += tmpLeft2 * tmpRight1;
            tmp31 += tmpLeft3 * tmpRight1;
            tmp41 += tmpLeft4 * tmpRight1;
            tmp02 += tmpLeft0 * tmpRight2;
            tmp12 += tmpLeft1 * tmpRight2;
            tmp22 += tmpLeft2 * tmpRight2;
            tmp32 += tmpLeft3 * tmpRight2;
            tmp42 += tmpLeft4 * tmpRight2;
            tmp03 += tmpLeft0 * tmpRight3;
            tmp13 += tmpLeft1 * tmpRight3;
            tmp23 += tmpLeft2 * tmpRight3;
            tmp33 += tmpLeft3 * tmpRight3;
            tmp43 += tmpLeft4 * tmpRight3;
            tmp04 += tmpLeft0 * tmpRight4;
            tmp14 += tmpLeft1 * tmpRight4;
            tmp24 += tmpLeft2 * tmpRight4;
            tmp34 += tmpLeft3 * tmpRight4;
            tmp44 += tmpLeft4 * tmpRight4;
        }

        product[0] = tmp00;
        product[1] = tmp10;
        product[2] = tmp20;
        product[3] = tmp30;
        product[4] = tmp40;
        product[5] = tmp01;
        product[6] = tmp11;
        product[7] = tmp21;
        product[8] = tmp31;
        product[9] = tmp41;
        product[10] = tmp02;
        product[11] = tmp12;
        product[12] = tmp22;
        product[13] = tmp32;
        product[14] = tmp42;
        product[15] = tmp03;
        product[16] = tmp13;
        product[17] = tmp23;
        product[18] = tmp33;
        product[19] = tmp43;
        product[20] = tmp04;
        product[21] = tmp14;
        product[22] = tmp24;
        product[23] = tmp34;
        product[24] = tmp44;
    }

    static void fill6xN(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        int tmpRowDim = 6;
        int tmpColDim = product.length / tmpRowDim;

        for (int j = 0; j < tmpColDim; j++) {

            double tmp0J = 0.0;
            double tmp1J = 0.0;
            double tmp2J = 0.0;
            double tmp3J = 0.0;
            double tmp4J = 0.0;
            double tmp5J = 0.0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                double tmpRightCJ = right.doubleValue(Structure2D.index(complexity, c, j));
                tmp0J += left[tmpIndex++] * tmpRightCJ;
                tmp1J += left[tmpIndex++] * tmpRightCJ;
                tmp2J += left[tmpIndex++] * tmpRightCJ;
                tmp3J += left[tmpIndex++] * tmpRightCJ;
                tmp4J += left[tmpIndex++] * tmpRightCJ;
                tmp5J += left[tmpIndex++] * tmpRightCJ;
            }

            product[tmpIndex = j * tmpRowDim] = tmp0J;
            product[++tmpIndex] = tmp1J;
            product[++tmpIndex] = tmp2J;
            product[++tmpIndex] = tmp3J;
            product[++tmpIndex] = tmp4J;
            product[++tmpIndex] = tmp5J;
        }
    }

    static void fill7xN(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        int tmpRowDim = 7;
        int tmpColDim = product.length / tmpRowDim;

        for (int j = 0; j < tmpColDim; j++) {

            double tmp0J = 0.0;
            double tmp1J = 0.0;
            double tmp2J = 0.0;
            double tmp3J = 0.0;
            double tmp4J = 0.0;
            double tmp5J = 0.0;
            double tmp6J = 0.0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                double tmpRightCJ = right.doubleValue(Structure2D.index(complexity, c, j));
                tmp0J += left[tmpIndex++] * tmpRightCJ;
                tmp1J += left[tmpIndex++] * tmpRightCJ;
                tmp2J += left[tmpIndex++] * tmpRightCJ;
                tmp3J += left[tmpIndex++] * tmpRightCJ;
                tmp4J += left[tmpIndex++] * tmpRightCJ;
                tmp5J += left[tmpIndex++] * tmpRightCJ;
                tmp6J += left[tmpIndex++] * tmpRightCJ;
            }

            product[tmpIndex = j * tmpRowDim] = tmp0J;
            product[++tmpIndex] = tmp1J;
            product[++tmpIndex] = tmp2J;
            product[++tmpIndex] = tmp3J;
            product[++tmpIndex] = tmp4J;
            product[++tmpIndex] = tmp5J;
            product[++tmpIndex] = tmp6J;
        }
    }

    static void fill8xN(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        int tmpRowDim = 8;
        int tmpColDim = product.length / tmpRowDim;

        for (int j = 0; j < tmpColDim; j++) {

            double tmp0J = 0.0;
            double tmp1J = 0.0;
            double tmp2J = 0.0;
            double tmp3J = 0.0;
            double tmp4J = 0.0;
            double tmp5J = 0.0;
            double tmp6J = 0.0;
            double tmp7J = 0.0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                double tmpRightCJ = right.doubleValue(Structure2D.index(complexity, c, j));
                tmp0J += left[tmpIndex++] * tmpRightCJ;
                tmp1J += left[tmpIndex++] * tmpRightCJ;
                tmp2J += left[tmpIndex++] * tmpRightCJ;
                tmp3J += left[tmpIndex++] * tmpRightCJ;
                tmp4J += left[tmpIndex++] * tmpRightCJ;
                tmp5J += left[tmpIndex++] * tmpRightCJ;
                tmp6J += left[tmpIndex++] * tmpRightCJ;
                tmp7J += left[tmpIndex++] * tmpRightCJ;
            }

            product[tmpIndex = j * tmpRowDim] = tmp0J;
            product[++tmpIndex] = tmp1J;
            product[++tmpIndex] = tmp2J;
            product[++tmpIndex] = tmp3J;
            product[++tmpIndex] = tmp4J;
            product[++tmpIndex] = tmp5J;
            product[++tmpIndex] = tmp6J;
            product[++tmpIndex] = tmp7J;
        }
    }

    static void fill9xN(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        int tmpRowDim = 9;
        int tmpColDim = product.length / tmpRowDim;

        for (int j = 0; j < tmpColDim; j++) {

            double tmp0J = 0.0;
            double tmp1J = 0.0;
            double tmp2J = 0.0;
            double tmp3J = 0.0;
            double tmp4J = 0.0;
            double tmp5J = 0.0;
            double tmp6J = 0.0;
            double tmp7J = 0.0;
            double tmp8J = 0.0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                double tmpRightCJ = right.doubleValue(Structure2D.index(complexity, c, j));
                tmp0J += left[tmpIndex++] * tmpRightCJ;
                tmp1J += left[tmpIndex++] * tmpRightCJ;
                tmp2J += left[tmpIndex++] * tmpRightCJ;
                tmp3J += left[tmpIndex++] * tmpRightCJ;
                tmp4J += left[tmpIndex++] * tmpRightCJ;
                tmp5J += left[tmpIndex++] * tmpRightCJ;
                tmp6J += left[tmpIndex++] * tmpRightCJ;
                tmp7J += left[tmpIndex++] * tmpRightCJ;
                tmp8J += left[tmpIndex++] * tmpRightCJ;
            }

            product[tmpIndex = j * tmpRowDim] = tmp0J;
            product[++tmpIndex] = tmp1J;
            product[++tmpIndex] = tmp2J;
            product[++tmpIndex] = tmp3J;
            product[++tmpIndex] = tmp4J;
            product[++tmpIndex] = tmp5J;
            product[++tmpIndex] = tmp6J;
            product[++tmpIndex] = tmp7J;
            product[++tmpIndex] = tmp8J;
        }
    }

    static void fillMx1(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {
        Arrays.fill(product, 0D);
        MultiplyRight.addMx1(product, left, complexity, right);
    }

    static void fillMx1(final float[] product, final float[] left, final int complexity, final Access1D<?> right) {
        Arrays.fill(product, 0F);
        MultiplyRight.addMx1(product, left, complexity, right);
    }

    static void fillMxN(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        Arrays.fill(product, 0D);

        MultiplyRight.addMxC(product, 0, Math.toIntExact(right.size() / complexity), left, complexity, right);
    }

    static void fillMxN(final float[] product, final float[] left, final int complexity, final Access1D<?> right) {

        Arrays.fill(product, 0F);

        MultiplyRight.addMxC(product, 0, Math.toIntExact(right.size() / complexity), left, complexity, right);
    }

    static void fillMxN_MT(final double[] product, final double[] left, final int complexity, final Access1D<?> right) {

        Arrays.fill(product, 0D);

        MultiplyRight.addMxN_MT(product, left, complexity, right);
    }

    static void fillMxN_MT(final float[] product, final float[] left, final int complexity, final Access1D<?> right) {

        Arrays.fill(product, 0F);

        MultiplyRight.addMxN_MT(product, left, complexity, right);
    }

}
