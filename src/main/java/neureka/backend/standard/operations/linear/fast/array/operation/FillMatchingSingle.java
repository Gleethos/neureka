/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array.operation;

public final class FillMatchingSingle {

    public static void fill(final double[] data, final double[] values) {
        int limit = Math.min(data.length, values.length);
        for (int i = 0; i < limit; i++) {
            data[i] = values[i];
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
