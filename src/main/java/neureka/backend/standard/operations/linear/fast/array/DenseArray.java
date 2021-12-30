/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.structure.FactorySupplement;

import java.util.ArrayList;

/**
 * <p>
 * Each and every element occupies memory and holds a value.
 * </p>
 *
 */
public abstract class DenseArray<N extends Comparable<N>> extends BasicArray<N> {

    public static abstract
            class Factory<N extends Comparable<N>>
            extends ArrayFactory<N, DenseArray<N>>
            implements FactorySupplement {

    }

    /**
     * Exists as a private constant in {@link ArrayList}. The Oracle JVM seems to actually be limited at
     * Integer.MAX_VALUE - 2, but other JVM:s may have different limits.
     */
    public static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

    protected DenseArray(final Factory<N> factory) {
        super(factory);
    }

}
