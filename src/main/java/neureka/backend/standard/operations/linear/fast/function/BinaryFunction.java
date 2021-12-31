/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

import neureka.backend.standard.operations.linear.fast.type.NumberDefinition;

import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;

public interface BinaryFunction<N extends Comparable<N>> extends BinaryOperator<N>, DoubleBinaryOperator {

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

    double invoke(double arg1, double arg2);

    default float invoke(final float arg1, final float arg2) {
        return (float) this.invoke((double) arg1, (double) arg2);
    }

    N invoke(N arg1, N arg2);

}
