/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array.operation;

import neureka.backend.standard.operations.linear.fast.array.F32Array;
import neureka.backend.standard.operations.linear.fast.array.F64Array;
import neureka.backend.standard.operations.linear.fast.function.BinaryFunction;
import neureka.backend.standard.operations.linear.fast.function.FunMath;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;

public final class OperationBinary {

    public static int THRESHOLD = 256;

    public static void invoke(final double[] data, final int first, final int limit, final int step, final Access1D<Double> left,
            final BinaryFunction<Double> function, final Access1D<Double> right) {
        if (left instanceof F64Array && right instanceof F64Array) {
            OperationBinary.invoke(data, first, limit, step, ((F64Array) left).data, function, ((F64Array) right).data);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left.doubleValue(i), right.doubleValue(i));
            }
        }
    }

    public static void invoke(final double[] data, final int first, final int limit, final int step, final double left, final BinaryFunction<Double> function,
            final double[] right) {
        if (function == FunMath.ADD) {
            PrimitiveArrayOperation.add(data, first, limit, step, left, right);
        } else if (function == FunMath.DIVIDE) {
            PrimitiveArrayOperation.divide(data, first, limit, step, left, right);
        } else if (function == FunMath.MULTIPLY) {
            PrimitiveArrayOperation.multiply(data, first, limit, step, left, right);
        } else if (function == FunMath.SUBTRACT) {
            PrimitiveArrayOperation.subtract(data, first, limit, step, left, right);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left, right[i]);
            }
        }
    }

    public static void invoke(final double[] data, final int first, final int limit, final int step, final double[] left, final BinaryFunction<Double> function,
            final double right) {
        if (function == FunMath.ADD) {
            PrimitiveArrayOperation.add(data, first, limit, step, left, right);
        } else if (function == FunMath.DIVIDE) {
            PrimitiveArrayOperation.divide(data, first, limit, step, left, right);
        } else if (function == FunMath.MULTIPLY) {
            PrimitiveArrayOperation.multiply(data, first, limit, step, left, right);
        } else if (function == FunMath.SUBTRACT) {
            PrimitiveArrayOperation.subtract(data, first, limit, step, left, right);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left[i], right);
            }
        }
    }

    public static void invoke(final float[] data, final int first, final int limit, final int step, final Access1D<Double> left,
            final BinaryFunction<Double> function, final Access1D<Double> right) {
        if (left instanceof F32Array && right instanceof F32Array) {
            OperationBinary.invoke(data, first, limit, step, ((F32Array) left).data, function, ((F32Array) right).data);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left.floatValue(i), right.floatValue(i));
            }
        }
    }

    public static void invoke(final float[] data, final int first, final int limit, final int step, final Access1D<Double> left,
            final BinaryFunction<Double> function, final float right) {
        if (left instanceof F32Array) {
            OperationBinary.invoke(data, first, limit, step, ((F32Array) left).data, function, right);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left.floatValue(i), right);
            }
        }
    }

    public static void invoke(final float[] data, final int first, final int limit, final int step, final float left, final BinaryFunction<Double> function,
            final float[] right) {
        if (function == FunMath.ADD) {
            PrimitiveArrayOperation.add(data, first, limit, step, left, right);
        } else if (function == FunMath.DIVIDE) {
            PrimitiveArrayOperation.divide(data, first, limit, step, left, right);
        } else if (function == FunMath.MULTIPLY) {
            PrimitiveArrayOperation.multiply(data, first, limit, step, left, right);
        } else if (function == FunMath.SUBTRACT) {
            PrimitiveArrayOperation.subtract(data, first, limit, step, left, right);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left, right[i]);
            }
        }
    }

    public static void invoke(final float[] data, final int first, final int limit, final int step, final float[] left, final BinaryFunction<Double> function,
            final float right) {
        if (function == FunMath.ADD) {
            PrimitiveArrayOperation.add(data, first, limit, step, left, right);
        } else if (function == FunMath.DIVIDE) {
            PrimitiveArrayOperation.divide(data, first, limit, step, left, right);
        } else if (function == FunMath.MULTIPLY) {
            PrimitiveArrayOperation.multiply(data, first, limit, step, left, right);
        } else if (function == FunMath.SUBTRACT) {
            PrimitiveArrayOperation.subtract(data, first, limit, step, left, right);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left[i], right);
            }
        }
    }

    public static <N extends Comparable<N>> void invoke(final N[] data, final int first, final int limit, final int step, final Access1D<N> left,
            final BinaryFunction<N> function, final Access1D<N> right) {
        for (int i = first; i < limit; i += step) {
            data[i] = function.invoke(left.get(i), right.get(i));
        }
    }

    public static <N extends Comparable<N>> void invoke(final N[] data, final int first, final int limit, final int step, final Access1D<N> left,
            final BinaryFunction<N> function, final N right) {
        for (int i = first; i < limit; i += step) {
            data[i] = function.invoke(left.get(i), right);
        }
    }

    public static <N extends Comparable<N>> void invoke(final N[] data, final int first, final int limit, final int step, final N left,
            final BinaryFunction<N> function, final Access1D<N> right) {
        for (int i = first; i < limit; i += step) {
            data[i] = function.invoke(left, right.get(i));
        }
    }

    static void invoke(final double[] data, final int first, final int limit, final int step, final double[] left, final BinaryFunction<Double> function,
            final double[] right) {
        if (function == FunMath.ADD) {
            PrimitiveArrayOperation.add(data, first, limit, step, left, right);
        } else if (function == FunMath.DIVIDE) {
            PrimitiveArrayOperation.divide(data, first, limit, step, left, right);
        } else if (function == FunMath.MULTIPLY) {
            PrimitiveArrayOperation.multiply(data, first, limit, step, left, right);
        } else if (function == FunMath.SUBTRACT) {
            PrimitiveArrayOperation.subtract(data, first, limit, step, left, right);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left[i], right[i]);
            }
        }
    }

    static void invoke(final float[] data, final int first, final int limit, final int step, final float[] left, final BinaryFunction<Double> function,
            final float[] right) {
        if (function == FunMath.ADD) {
            PrimitiveArrayOperation.add(data, first, limit, step, left, right);
        } else if (function == FunMath.DIVIDE) {
            PrimitiveArrayOperation.divide(data, first, limit, step, left, right);
        } else if (function == FunMath.MULTIPLY) {
            PrimitiveArrayOperation.multiply(data, first, limit, step, left, right);
        } else if (function == FunMath.SUBTRACT) {
            PrimitiveArrayOperation.subtract(data, first, limit, step, left, right);
        } else {
            for (int i = first; i < limit; i += step) {
                data[i] = function.invoke(left[i], right[i]);
            }
        }
    }

}
