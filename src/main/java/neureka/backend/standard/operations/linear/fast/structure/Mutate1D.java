/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

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

    void set(long index, double value);

}
