/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array.operation;

import neureka.backend.standard.operations.linear.fast.function.ParameterFunction;

public final class OperationParameter {

    public static int THRESHOLD = 256;

    public static void invoke(final double[] data, final int first, final int limit, final int step, final double[] values,
            final ParameterFunction<Double> function, final int param) {
        for (int i = first; i < limit; i += step) {
            data[i] = function.invoke(values[i], param);
        }
    }

    public static void invoke(final float[] data, final int first, final int limit, final int step, final float[] values,
            final ParameterFunction<Double> function, final int param) {
        for (int i = first; i < limit; i += step) {
            data[i] = function.invoke(values[i], param);
        }
    }

}
