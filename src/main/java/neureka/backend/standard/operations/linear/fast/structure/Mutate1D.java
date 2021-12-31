/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

import neureka.backend.standard.operations.linear.fast.function.BinaryFunction;
import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;

/**
 * 1-dimensional mutator methods
 *
 */
public interface Mutate1D extends Size {

    /**
     * Fills the target
     *
     * 
     */
    interface Fillable<N extends Comparable<N>> extends Size {

        /**
         * <p>
         * Will fill the elements of [this] with the corresponding input values, and in the process (if
         * necessary) convert the elements to the correct type:
         * </p>
         * <code>this(i) = values(i)</code>
         */
        default void fillMatchingAccess(final Access1D<?> values) {
            //Structure1D.loopMatching(this, values, i -> this.fillOneAccess(i, values, i));
        }

        void fillOneAccess(long index, Access1D<?> values, long valueIndex);

    }

    interface Modifiable<N extends Comparable<N>> extends Size {

        void operateAll(final UnaryFunction<N> modifier);

        void operateWith(final Access1D<N> left, final BinaryFunction<N> function);

        void operate(final BinaryFunction<N> function, final Access1D<N> right);

    }

    void set(long index, double value);

}
