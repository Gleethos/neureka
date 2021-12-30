/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array.operation;

import neureka.backend.standard.operations.linear.fast.array.F32Array;
import neureka.backend.standard.operations.linear.fast.array.F64Array;
import neureka.backend.standard.operations.linear.fast.function.BinaryFunction.FixedFirst;
import neureka.backend.standard.operations.linear.fast.function.BinaryFunction.FixedSecond;
import neureka.backend.standard.operations.linear.fast.function.FunMath;
import neureka.backend.standard.operations.linear.fast.function.ParameterFunction.FixedParameter;
import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;

public final class OperationUnary {

    public static int THRESHOLD = 256;

    public static void invoke(final double[] data, final int first, final int limit, final int step, final Access1D<Double> values,
            final UnaryFunction<Double> function) {
        if (values instanceof F64Array) {
            OperationUnary.invoke(data, first, limit, step, ((F64Array) values).data, function);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(values.doubleValue(i));
            }
        }
    }

    public static void invoke(final float[] data, final int first, final int limit, final int step, final Access1D<Double> values,
            final UnaryFunction<Double> function) {
        if (values instanceof F32Array) {
            OperationUnary.invoke(data, first, limit, step, ((F32Array) values).data, function);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(values.floatValue(i));
            }
        }
    }

    public static void invoke(final float[] data, final int first, final int limit, final int step, final float[] values,
            final UnaryFunction<Double> function) {
        if (function == FunMath.NEGATE) {
            PrimitiveArrayOperation.negate(data, first, limit, step, values);
        } else if (function instanceof FixedFirst<?>) {
            final FixedFirst<Double> tmpFunc = (FixedFirst<Double>) function;
            OperationBinary.invoke(data, first, limit, step, tmpFunc.floatValue(), tmpFunc.getFunction(), values);
        } else if (function instanceof FixedSecond<?>) {
            final FixedSecond<Double> tmpFunc = (FixedSecond<Double>) function;
            OperationBinary.invoke(data, first, limit, step, values, tmpFunc.getFunction(), tmpFunc.floatValue());
        } else if (function instanceof FixedParameter<?>) {
            final FixedParameter<Double> tmpFunc = (FixedParameter<Double>) function;
            OperationParameter.invoke(data, first, limit, step, values, tmpFunc.getFunction(), tmpFunc.getParameter());
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(values[i]);
            }
        }
    }

    public static <N extends Comparable<N>> void invoke(final N[] data, final int first, final int limit, final int step, final Access1D<N> value,
            final UnaryFunction<N> function) {
        for (int i = first; i < limit; i += step) {
            data[i] = function.invoke(value.get(i));
        }
    }

    static void invoke(final double[] data, final int first, final int limit, final int step, final double[] values, final UnaryFunction<Double> function) {
        if (function == FunMath.NEGATE) {
            PrimitiveArrayOperation.negate(data, first, limit, step, values);
        } else if (function instanceof FixedFirst<?>) {
            final FixedFirst<Double> tmpFunc = (FixedFirst<Double>) function;
            OperationBinary.invoke(data, first, limit, step, tmpFunc.doubleValue(), tmpFunc.getFunction(), values);
        } else if (function instanceof FixedSecond<?>) {
            final FixedSecond<Double> tmpFunc = (FixedSecond<Double>) function;
            OperationBinary.invoke(data, first, limit, step, values, tmpFunc.getFunction(), tmpFunc.doubleValue());
        } else if (function instanceof FixedParameter<?>) {
            final FixedParameter<Double> tmpFunc = (FixedParameter<Double>) function;
            OperationParameter.invoke(data, first, limit, step, values, tmpFunc.getFunction(), tmpFunc.getParameter());
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(values[i]);
            }
        }
    }

}
