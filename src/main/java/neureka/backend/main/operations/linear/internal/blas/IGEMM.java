/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
package neureka.backend.main.operations.linear.internal.blas;

import neureka.devices.host.CPU;

/**
 *  A collection of primitive sub-routines for matrix multiplication performed on
 *  continuous arrays which are designed so that they can be vectorized by the
 *  JVMs JIT compiler (AVX instructions).
 */
public class IGEMM {

    @FunctionalInterface
    public interface VectorOperationI32 {
        void invoke(
                int[] product,
                int[] left,
                int complexity,
                int[] right
        );
    }

    @FunctionalInterface
    public interface VectorOperationI64 {
        void invoke(
                long[] product,
                long[] left,
                int complexity,
                long[] right
        );
    }

    public static VectorOperationI32 operationForI32(
            boolean rowMajor, final long rows, final long columns
    ) {
        if ( rows > CPU.PARALLELIZATION_THRESHOLD && columns > CPU.PARALLELIZATION_THRESHOLD) {
            return ( rowMajor
                    ? IGEMM::threaded_I32_MxN_RM
                    : IGEMM::threaded_I32_MxN_CM
            );
        }
        if ( columns == 1 ) {
            return ( rowMajor
                    ? IGEMM::full_I32_Mx1_RM
                    : IGEMM::full_I32_Mx1_CM
            );
        }
        if ( rows == 1 )
            return (
                    rowMajor
                            ? IGEMM::full_I32_1xN_RM
                            : IGEMM::full_I32_1xN_CM
            );
        return (
                rowMajor
                        ? IGEMM::full_I32_MxN_RM
                        : IGEMM::full_I32_MxN_CM
        );
    }

    public static VectorOperationI64 operationForI64(
            boolean rowMajor, final long rows, final long columns
    ) {
        if ( rows > CPU.PARALLELIZATION_THRESHOLD && columns > CPU.PARALLELIZATION_THRESHOLD )
            return ( rowMajor
                    ? IGEMM::threaded_I64_MxN_RM
                    : IGEMM::threaded_I64_MxN_CM
            );

        if ( !rowMajor ) { // Supported in column major only!
            if (rows == 5 && columns == 5) return IGEMM::full_I64_5x5_CM;
            if (rows == 4 && columns == 4) return IGEMM::full_I64_4x4_CM;
            if (rows == 3 && columns == 3) return IGEMM::full_I64_3x3_CM;
            if (rows == 2 && columns == 2) return IGEMM::full_I64_2x2_CM;
            if (rows == 1 && columns == 1) return IGEMM::full_I64_1x1_CM;
        }
        if ( columns == 1 )
            return (
                    rowMajor
                            ? IGEMM::full_I64_Mx1_RM
                            : IGEMM::full_I64_Mx1_CM
            );

        if ( !rowMajor ) { // Supported in column major only!
            if ( rows == 10) return IGEMM::full_I64_0xN_CM;
            if ( rows == 9 ) return IGEMM::full_I64_9xN_CM;
            if ( rows == 8 ) return IGEMM::full_I64_8xN_CM;
            if ( rows == 7 ) return IGEMM::full_I64_7xN_CM;
            if ( rows == 6 ) return IGEMM::full_I64_6xN_CM;
        }
        if ( rows == 1 )
            return (
                    rowMajor
                            ? IGEMM::full_I64_1xN_RM
                            : IGEMM::full_I64_1xN_CM
            );

        return (
                rowMajor
                        ? IGEMM::full_I64_MxN_RM
                        : IGEMM::full_I64_MxN_CM
        );
    }

    static void full_I64_Mx1_CM(
            final long[] product,
            final long[] left,
            final int colCount,
            final long[] right
    ) {
        int rowCount = product.length;
        for ( int c = 0; c < colCount; c++ ) {
            IAXPY.invoke(
                    product,
                    0,
                    right[c],
                    left,
                    c * rowCount,
                    0,
                    rowCount
            );
        }
    }

    static void full_I64_Mx1_RM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        int colCount = right.length;
        int leftRowCount = left.length / complexity;
        for (int ri = 0; ri < leftRowCount; ri++) {
            product[ri] =
                    IDOT.invoke(
                            left,
                            ri * complexity,
                            right,
                            0,
                            0,
                            colCount
                    );
        }
    }

    static void full_I32_Mx1_CM(
            final int[] product,
            final int[] left,
            final int complexity,
            final int[] right
    ) {
        int nbRows = product.length;

        for (int c = 0; c < complexity; c++) {
            IAXPY.invoke(
                    product,
                    0,
                    right[c],
                    left,
                    c * nbRows,
                    0, nbRows
            );
        }
    }

    static void full_I32_Mx1_RM(
            final int[] product,
            final int[] left,
            final int complexity,
            final int[] right
    ) {
        int colCount = right.length;
        int leftRowCount = left.length / complexity;
        for ( int ri = 0; ri < leftRowCount; ri++ ) {
            product[ri] =
                    IDOT.invoke(
                            left,
                            ri * complexity,
                            right,
                            0,
                            0,
                            colCount
                    );
        }
    }

    static void partial_I64_MxN_CM(
            final long[] product,
            final int firstColumn,
            final int columnLimit,
            final long[] left,
            final int commonColumnCount,
            final long[] right
    ) {
        int leftRowCount = left.length / commonColumnCount;

        for ( int ci = 0; ci < commonColumnCount; ci++ ) {
            for ( int j = firstColumn; j < columnLimit; j++ ) {
                IAXPY.invoke(
                        product,
                        j * leftRowCount,
                        right[ci + j * commonColumnCount],
                        left,
                        ci * leftRowCount,
                        0,
                        leftRowCount
                );
            }
        }
    }

    static void partial_I64_MxN_RM(
            final long[] product,
            final int firstRow,
            final int rowLimit,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        int rightCols = right.length / complexity;
        for ( int c = 0; c < complexity; c++ ) {
            for ( int j = firstRow; j < rowLimit; j++ ) {
                IAXPY.invoke(
                        product,
                        j * rightCols,
                        left[c + j * complexity],
                        right,
                        c * rightCols,
                        0,
                        rightCols
                );
            }
        }
    }

    static void partial_I32_MxN_CM(
            final int[] product,
            final int firstColumn, // right column
            final int columnLimit,
            final int[] left,
            final int complexity,
            final int[] right
    ) {
        int nbRows = left.length / complexity;
        for ( int c = 0; c < complexity; c++ ) {
            for ( int j = firstColumn; j < columnLimit; j++ ) {
                IAXPY.invoke(
                        product,
                        j * nbRows,
                        right[c + j * complexity],
                        left,
                        c * nbRows,
                        0,
                        nbRows
                );
            }
        }
    }

    static void partial_I32_MxN_RM(
            final int[] product,
            final int firstRow, // left row
            final int rowLimit,
            final int[] left,
            final int complexity,
            final int[] right
    ) {
        int rightCols = right.length / complexity;
        for ( int c = 0; c < complexity; c++ ) {
            for ( int j = firstRow; j < rowLimit; j++ ) {
                IAXPY.invoke(
                        product,
                        j * rightCols,
                        left[c + j * complexity],
                        right,
                        c * rightCols,
                        0,
                        rightCols
                );
            }
        }
    }


    static void threaded_I64_MxN_CM(final long[] product, final long[] left, final int complexity, final long[] right) {
        CPU.get()
                .getExecutor()
                .threaded(
                        0,
                        right.length / complexity,
                        (f, l) -> IGEMM.partial_I64_MxN_CM(product, f, l, left, complexity, right)
                );
    }

    static void threaded_I32_MxN_CM(final int[] product, final int[] left, final int complexity, final int[] right) {
        CPU.get()
                .getExecutor()
                .threaded(
                        0,
                        right.length / complexity,
                        (f, l) -> IGEMM.partial_I32_MxN_CM(product, f, l, left, complexity, right)
                );
    }

    static void threaded_I32_MxN_RM(final int[] product, final int[] left, final int complexity, final int[] right) {
        CPU.get()
                .getExecutor()
                .threaded(
                        0,
                        left.length / complexity,
                        (f, l) -> IGEMM.partial_I32_MxN_RM(product, f, l, left, complexity, right)
                );
    }

    static void threaded_I64_MxN_RM(final long[] product, final long[] left, final int complexity, final long[] right) {
        CPU.get()
                .getExecutor()
                .threaded(
                        0,
                        left.length / complexity,
                        (f, l) -> IGEMM.partial_I64_MxN_RM(product, f, l, left, complexity, right)
                );
    }

    static void full_I64_0xN_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {

        int tmpRowDim = 10;
        int tmpColDim = right.length / complexity;

        for ( int j = 0; j < tmpColDim; j++ ) {

            long tmp0J = 0;
            long tmp1J = 0;
            long tmp2J = 0;
            long tmp3J = 0;
            long tmp4J = 0;
            long tmp5J = 0;
            long tmp6J = 0;
            long tmp7J = 0;
            long tmp8J = 0;
            long tmp9J = 0;

            int tmpIndex = 0;
            for ( int c = 0; c < complexity; c++ ) {
                long tmpRightCJ = right[c + j * complexity];
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

    static void full_I64_1x1_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        long tmp00 = 0;

        int nbRows = left.length / complexity;

        for ( int c = 0; c < complexity; c++ )
            tmp00 += left[c * nbRows] * right[c];

        product[0] = tmp00;
    }

    static void full_I64_1xN_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        for ( int j = 0, nbCols = product.length; j < nbCols; j++ )
            product[j] = IDOT.invoke(left, 0, right, j * complexity, 0, complexity);
    }

    static void full_I64_1xN_RM(
            final long[] product,
            final long[] left,
            final int rowCount,
            final long[] right
    ) {//...
        int colCount = product.length;
        for ( int ci = 0; ci < colCount; ci++ ) {
            IAXPY.invoke(
                    product,
                    0,
                    left[ci],
                    right,
                    ci * rowCount,
                    0,
                    rowCount
            );
        }
    }

    static void full_I32_1xN_CM(
            final int[] product,
            final int[] left,
            final int complexity,
            final int[] right
    ) {
        for (int j = 0, nbCols = product.length; j < nbCols; j++) {
            product[j] = IDOT.invoke(left, 0, right, j * complexity, 0, complexity);
        }
    }


    static void full_I32_1xN_RM(
            final int[] product,
            final int[] left,
            final int rowCount,
            final int[] right
    ) {//...
        int colCount = product.length;
        for (int ci = 0; ci < colCount; ci++) {
            IAXPY.invoke(
                    product,
                    0,
                    left[ci],
                    right,
                    ci * rowCount,
                    0,
                    rowCount
            );
        }
    }


    static void full_I64_2x2_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        long tmp00 = 0;
        long tmp10 = 0;
        long tmp01 = 0;
        long tmp11 = 0;

        int tmpIndex;
        for (int c = 0; c < complexity; c++) {

            tmpIndex = c * 2;
            long tmpLeft0 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft1 = left[tmpIndex];
            tmpIndex = c;
            long tmpRight0 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight1 = right[tmpIndex];

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

    static void full_I64_3x3_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        long tmp00 = 0;
        long tmp10 = 0;
        long tmp20 = 0;
        long tmp01 = 0;
        long tmp11 = 0;
        long tmp21 = 0;
        long tmp02 = 0;
        long tmp12 = 0;
        long tmp22 = 0;

        int tmpIndex;
        for (int c = 0; c < complexity; c++) {

            tmpIndex = c * 3;
            long tmpLeft0 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft1 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft2 = left[tmpIndex];
            tmpIndex = c;
            long tmpRight0 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight1 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight2 = right[tmpIndex];

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

    static void full_I64_4x4_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        long tmp00 = 0;
        long tmp10 = 0;
        long tmp20 = 0;
        long tmp30 = 0;
        long tmp01 = 0;
        long tmp11 = 0;
        long tmp21 = 0;
        long tmp31 = 0;
        long tmp02 = 0;
        long tmp12 = 0;
        long tmp22 = 0;
        long tmp32 = 0;
        long tmp03 = 0;
        long tmp13 = 0;
        long tmp23 = 0;
        long tmp33 = 0;

        int tmpIndex;
        for (int c = 0; c < complexity; c++) {

            tmpIndex = c * 4;
            long tmpLeft0 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft1 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft2 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft3 = left[tmpIndex];
            tmpIndex = c;
            long tmpRight0 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight1 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight2 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight3 = right[tmpIndex];

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

    static void full_I64_5x5_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        long tmp00 = 0;
        long tmp10 = 0;
        long tmp20 = 0;
        long tmp30 = 0;
        long tmp40 = 0;
        long tmp01 = 0;
        long tmp11 = 0;
        long tmp21 = 0;
        long tmp31 = 0;
        long tmp41 = 0;
        long tmp02 = 0;
        long tmp12 = 0;
        long tmp22 = 0;
        long tmp32 = 0;
        long tmp42 = 0;
        long tmp03 = 0;
        long tmp13 = 0;
        long tmp23 = 0;
        long tmp33 = 0;
        long tmp43 = 0;
        long tmp04 = 0;
        long tmp14 = 0;
        long tmp24 = 0;
        long tmp34 = 0;
        long tmp44 = 0;

        int tmpIndex;
        for (int c = 0; c < complexity; c++) {

            tmpIndex = c * 5;
            long tmpLeft0 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft1 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft2 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft3 = left[tmpIndex];
            tmpIndex++;
            long tmpLeft4 = left[tmpIndex];
            tmpIndex = c;
            long tmpRight0 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight1 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight2 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight3 = right[tmpIndex];
            tmpIndex += complexity;
            long tmpRight4 = right[tmpIndex];

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

    static void full_I64_6xN_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {

        int tmpRowDim = 6;
        int tmpColDim = right.length / complexity;

        for (int j = 0; j < tmpColDim; j++) {

            long tmp0J = 0;
            long tmp1J = 0;
            long tmp2J = 0;
            long tmp3J = 0;
            long tmp4J = 0;
            long tmp5J = 0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                long tmpRightCJ = right[c + j * complexity];
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

    static void full_I64_7xN_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {

        int tmpRowDim = 7;
        int tmpColDim = right.length / complexity;

        for (int j = 0; j < tmpColDim; j++) {

            long tmp0J = 0;
            long tmp1J = 0;
            long tmp2J = 0;
            long tmp3J = 0;
            long tmp4J = 0;
            long tmp5J = 0;
            long tmp6J = 0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                long tmpRightCJ = right[c + j * complexity];
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

    static void full_I64_8xN_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {

        int tmpRowDim = 8;
        int tmpColDim = right.length / complexity;

        for (int j = 0; j < tmpColDim; j++) {

            long tmp0J = 0;
            long tmp1J = 0;
            long tmp2J = 0;
            long tmp3J = 0;
            long tmp4J = 0;
            long tmp5J = 0;
            long tmp6J = 0;
            long tmp7J = 0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                long tmpRightCJ = right[c + j * complexity];
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

    static void full_I64_9xN_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {

        int tmpRowDim = 9;
        int tmpColDim = right.length / complexity;

        for (int j = 0; j < tmpColDim; j++) {

            long tmp0J = 0;
            long tmp1J = 0;
            long tmp2J = 0;
            long tmp3J = 0;
            long tmp4J = 0;
            long tmp5J = 0;
            long tmp6J = 0;
            long tmp7J = 0;
            long tmp8J = 0;

            int tmpIndex = 0;
            for (int c = 0; c < complexity; c++) {
                long tmpRightCJ = right[c + j * complexity];
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

    static void full_I64_MxN_CM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        IGEMM.partial_I64_MxN_CM(product, 0, right.length / complexity, left, complexity, right);
    }

    static void full_I64_MxN_RM(
            final long[] product,
            final long[] left,
            final int complexity,
            final long[] right
    ) {
        IGEMM.partial_I64_MxN_RM(product, 0, left.length / complexity, left, complexity, right);
    }

    static void full_I32_MxN_CM(
            final int[] product,
            final int[] left,
            final int complexity,
            final int[] right
    ) {
        IGEMM.partial_I32_MxN_CM(product, 0, right.length / complexity, left, complexity, right);
    }

    static void full_I32_MxN_RM(
            final int[] product,
            final int[] left,
            final int complexity,
            final int[] right
    ) {
        IGEMM.partial_I32_MxN_RM(product, 0, left.length / complexity, left, complexity, right);
    }

}
