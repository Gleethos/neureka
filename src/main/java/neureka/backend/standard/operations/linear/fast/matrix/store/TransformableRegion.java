/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.structure.*;

/**
 * A transformable 2D (sub)region.
 *
 */
public interface TransformableRegion<N extends Comparable<N>>
        extends
            Mutate1D.Modifiable<N>,
            Mutate2D.Receiver<N>,
            Structure2D,
            Access2D<N>
{

    @FunctionalInterface
    interface FillByMultiplying<N extends Comparable<N>> {

        void invoke(TransformableRegion<N> product, Access1D<N> left, int complexity, Access1D<N> right);

        default void invoke(final TransformableRegion<N> product, final Access1D<N> left, final long complexity, final Access1D<N> right) {
            this.invoke(product, left, Math.toIntExact(complexity), right);
        }

    }

    void fillByMultiplying(final Access1D<N> left, final Access1D<N> right);

}
