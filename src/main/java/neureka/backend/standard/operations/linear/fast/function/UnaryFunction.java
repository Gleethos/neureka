/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

public interface UnaryFunction<N extends Comparable<N>> extends UnaryOperator<N>, DoubleUnaryOperator {

    default N apply(final N arg) {
        return this.invoke(arg);
    }

    default double applyAsDouble(final double arg) {
        return this.invoke(arg);
    }

    double invoke(double arg);

    default float invoke(final float arg) {
        return (float) this.invoke((double) arg);
    }

    N invoke(N arg);

    default short invoke(final short arg) {
        return (short) this.invoke((double) arg);
    }

}
