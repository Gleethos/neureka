/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

import neureka.backend.standard.operations.linear.fast.type.NumberDefinition;

import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;

public interface BinaryFunction<N extends Comparable<N>> extends BinaryOperator<N>, DoubleBinaryOperator {

    /**
     * A {@linkplain BinaryFunction} with a set/fixed first argument.
     *
     * 
     */
    final class FixedFirst<N extends Comparable<N>> implements UnaryFunction<N> {

        private final double myDoubleValue;
        private final float myFloatValue;
        private final BinaryFunction<N> myFunction;
        private final N myNumber;

        FixedFirst(final double arg1, final BinaryFunction<N> function) {

            super();

            myFunction = function;

            myNumber = (N) Double.valueOf(arg1);
            myDoubleValue = arg1;
            myFloatValue = (float) arg1;
        }

        FixedFirst(final N arg1, final BinaryFunction<N> function) {

            super();

            myFunction = function;

            myNumber = arg1;
            myDoubleValue = NumberDefinition.doubleValue(arg1);
            myFloatValue = NumberDefinition.floatValue(arg1);
        }

        public double doubleValue() {
            return myDoubleValue;
        }

        public float floatValue() {
            return myFloatValue;
        }

        public BinaryFunction<N> getFunction() {
            return myFunction;
        }

        public N getNumber() {
            return myNumber;
        }

        public double invoke(final double arg2) {
            return myFunction.invoke(myDoubleValue, arg2);
        }

        public float invoke(final float arg2) {
            return myFunction.invoke(myFloatValue, arg2);
        }

        public N invoke(final N arg2) {
            return myFunction.invoke(myNumber, arg2);
        }

    }

    /**
     * A {@linkplain BinaryFunction} with a set/fixed second argument.
     *
     * 
     */
    public static final class FixedSecond<N extends Comparable<N>> implements UnaryFunction<N> {

        private final double myDoubleValue;
        private final float myFloatValue;
        private final BinaryFunction<N> myFunction;
        private final N myNumber;

        FixedSecond(final BinaryFunction<N> function, final double arg2) {

            super();

            myFunction = function;

            myNumber = (N) Double.valueOf(arg2);
            myDoubleValue = arg2;
            myFloatValue = (float) arg2;
        }

        FixedSecond(final BinaryFunction<N> function, final N arg2) {

            super();

            myFunction = function;

            myNumber = arg2;
            myDoubleValue = NumberDefinition.doubleValue(arg2);
            myFloatValue = NumberDefinition.floatValue(arg2);
        }

        public double doubleValue() {
            return myDoubleValue;
        }

        public float floatValue() {
            return myFloatValue;
        }

        public BinaryFunction<N> getFunction() {
            return myFunction;
        }

        public N getNumber() {
            return myNumber;
        }

        public double invoke(final double arg1) {
            return myFunction.invoke(arg1, myDoubleValue);
        }

        public float invoke(final float arg1) {
            return myFunction.invoke(arg1, myFloatValue);
        }

        public N invoke(final N arg1) {
            return myFunction.invoke(arg1, myNumber);
        }

    }

    /**
     * @see java.util.function.BiFunction#apply(Object, Object)
     */
    default N apply(final N arg1, final N arg2) {
        return this.invoke(arg1, arg2);
    }

    /**
     * @see DoubleBinaryOperator#applyAsDouble(double, double)
     */
    default double applyAsDouble(final double arg1, final double arg2) {
        return this.invoke(arg1, arg2);
    }

    /**
     * To allow syntax like <code>array.modifyAll(DIVIDE.by(3.0));</code>
     *
     * @see #second(double)
     */
    default UnaryFunction<N> by(final double arg2) {
        return this.second(arg2);
    }

    /**
     * To allow syntax like <code>array.modifyAll(DIVIDE.by(3.0));</code>
     *
     * @see #second(double)
     */
    default UnaryFunction<N> by(final N arg2) {
        return this.second(arg2);
    }

    /**
     * @see #first(Comparable)
     */
    default UnaryFunction<N> first(final double arg1) {
        return new FixedFirst<>(arg1, this);
    }

    /**
     * Turns this binary function into a unary function with the first argument fixed/locked to the specified
     * value.
     *
     * @param arg1 The first argument of the binary function.
     * @return The resulting unary function.
     */
    default UnaryFunction<N> first(final N arg1) {
        return new FixedFirst<>(arg1, this);
    }

    default byte invoke(final byte arg1, final byte arg2) {
        return (byte) this.invoke((double) arg1, (double) arg2);
    }

    double invoke(double arg1, double arg2);

    default float invoke(final float arg1, final float arg2) {
        return (float) this.invoke((double) arg1, (double) arg2);
    }

    default int invoke(final int arg1, final int arg2) {
        return NumberDefinition.toInt(this.invoke((double) arg1, (double) arg2));
    }

    default long invoke(final long arg1, final long arg2) {
        return NumberDefinition.toLong(this.invoke((double) arg1, (double) arg2));
    }

    N invoke(N arg1, N arg2);

    default short invoke(final short arg1, final short arg2) {
        return (short) this.invoke((double) arg1, (double) arg2);
    }

    /**
     * @see #second(Comparable)
     */
    default UnaryFunction<N> second(final double arg2) {
        return new FixedSecond<>(this, arg2);
    }

    /**
     * Turns this binary function into a unary function with the second argument fixed/locked to the specified
     * value.
     *
     * @param arg2 The second argument of the binary function.
     * @return The resulting unary function.
     */
    default UnaryFunction<N> second(final N arg2) {
        return new FixedSecond<>(this, arg2);
    }

}
