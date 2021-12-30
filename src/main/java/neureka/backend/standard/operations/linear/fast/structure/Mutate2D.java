/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

/**
 * 2-dimensional mutator methods
 *
 */
public interface Mutate2D extends Structure2D, Mutate1D {

    interface Fillable<N extends Comparable<N>> extends Structure2D, Mutate1D.Fillable<N> {

        @Override
        default void fillOneAccess(final long index, final Access1D<?> values, final long valueIndex) {
            throw new IllegalStateException();
        }

    }

    interface Receiver<N extends Comparable<N>> extends Mutate2D, Fillable<N> {

    }

    @Override
    default void set(final long index, final double addend) {
        long structure = this.numberOfRows();
        this.set(Structure2D.row(index, structure), Structure2D.column(index, structure), addend);
    }

    void set(long row, long col, double value);

}
