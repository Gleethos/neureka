/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

import neureka.backend.standard.operations.linear.fast.structure.AccessScalar;

import java.util.function.DoubleSupplier;
import java.util.function.Supplier;

public interface NullaryFunction<N extends Comparable<N>> extends Supplier<N>, DoubleSupplier, AccessScalar<N> {

    double doubleValue();

    default N get() {
        return this.invoke();
    }

    default double getAsDouble() {
        return this.doubleValue();
    }

    N invoke();

}
