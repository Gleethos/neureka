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

/**
 * The ?dot routines perform a vector-vector reduction operation defined as Equation where xi and yi are
 * elements of vectors x and y.
 *
 */
public final class DOT {

    public static double invoke(final double[] array1, final int offset1, final double[] array2, final int offset2, final int first, final int limit) {
        return DOT.unrolled04(array1, offset1, array2, offset2, first, limit);
    }

    public static float invoke(final float[] array1, final int offset1, final float[] array2, final int offset2, final int first, final int limit) {
        return DOT.unrolled04(array1, offset1, array2, offset2, first, limit);
    }

    static double unrolled04(final double[] array1, final int offset1, final double[] array2, final int offset2, final int first, final int limit) {

        int remainder = (limit - first) % 4;

        double sum0 = 0F;
        double sum1 = 0F;
        double sum2 = 0F;
        double sum3 = 0F;

        int shift10 = offset1 + 0;
        int shift11 = offset1 + 1;
        int shift12 = offset1 + 2;
        int shift13 = offset1 + 3;

        int shift20 = offset2 + 0;
        int shift21 = offset2 + 1;
        int shift22 = offset2 + 2;
        int shift23 = offset2 + 3;

        int i = first;
        for (int lim = limit - remainder; i < lim; i += 4) {
            sum0 += array1[shift10 + i] * array2[shift20 + i];
            sum1 += array1[shift11 + i] * array2[shift21 + i];
            sum2 += array1[shift12 + i] * array2[shift22 + i];
            sum3 += array1[shift13 + i] * array2[shift23 + i];
        }
        for (; i < limit; i++) {
            sum0 += array1[shift10 + i] * array2[shift20 + i];
        }

        return sum0 + sum1 + sum2 + sum3;
    }

    static float unrolled04(final float[] array1, final int offset1, final float[] array2, final int offset2, final int first, final int limit) {

        int remainder = (limit - first) % 4;

        float sum0 = 0F;
        float sum1 = 0F;
        float sum2 = 0F;
        float sum3 = 0F;

        int shift10 = offset1 + 0;
        int shift11 = offset1 + 1;
        int shift12 = offset1 + 2;
        int shift13 = offset1 + 3;

        int shift20 = offset2 + 0;
        int shift21 = offset2 + 1;
        int shift22 = offset2 + 2;
        int shift23 = offset2 + 3;

        int i = first;
        for (int lim = limit - remainder; i < lim; i += 4) {
            sum0 += array1[shift10 + i] * array2[shift20 + i];
            sum1 += array1[shift11 + i] * array2[shift21 + i];
            sum2 += array1[shift12 + i] * array2[shift22 + i];
            sum3 += array1[shift13 + i] * array2[shift23 + i];
        }
        for (; i < limit; i++) {
            sum0 += array1[shift10 + i] * array2[shift20 + i];
        }

        return sum0 + sum1 + sum2 + sum3;
    }

}
