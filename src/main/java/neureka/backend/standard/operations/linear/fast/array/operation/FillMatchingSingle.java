/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array.operation;

import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.structure.Access2D;

public final class FillMatchingSingle {

    public static int THRESHOLD = 256;

    public static void copy(final double[] data, final int structure, final int firstColumn, final int limitColumn,
            final Access2D<? extends Comparable<?>> source) {
        int index = structure * firstColumn;
        for (int j = firstColumn; j < limitColumn; j++) {
            for (int i = 0; i < structure; i++) {
                data[index++] = source.doubleValue(i, j);
            }
        }
    }

    public static void copy(final float[] data, final int structure, final int firstColumn, final int limitColumn,
            final Access2D<? extends Comparable<?>> source) {
        int index = structure * firstColumn;
        for (int j = firstColumn; j < limitColumn; j++) {
            for (int i = 0; i < structure; i++) {
                data[index++] = source.floatValue(i, j);
            }
        }
    }

    public static void fill(final double[] data, final Access1D<?> values) {
        int limit = Math.min(data.length, values.size());
        for (int i = 0; i < limit; i++) {
            data[i] = values.doubleValue(i);
        }
    }

    public static void fill(final double[] data, final double[] values) {
        int limit = Math.min(data.length, values.length);
        for (int i = 0; i < limit; i++) {
            data[i] = values[i];
        }
    }

    public static void fill(final float[] data, final Access1D<?> values) {
        int limit = Math.min(data.length, values.size());
        for (int i = 0; i < limit; i++) {
            data[i] = values.floatValue(i);
        }
    }

    public static void fill(final float[] data, final float[] values) {
        int limit = Math.min(data.length, values.length);
        for (int i = 0; i < limit; i++) {
            data[i] = values[i];
        }
    }

    public static void invoke(final double[] source, final int sourceOffset, final double[] destination, final int destinationOffset, final int first,
            final int limit) {
        for (int i = first; i < limit; i++) {
            destination[destinationOffset + i] = source[sourceOffset + i];
        }
    }

}
