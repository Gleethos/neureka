/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.structure.*;

/**
 * A transformable 2D (sub)region.
 *
 */
public interface TransformableRegion<N extends Comparable<N>>
        extends
            Size,
            Mutate2D.Receiver<N>,
            Structure2D,
            Access2D<N>
{

    void fillByMultiplying(final Access1D<N> left, final Access1D<N> right);

}
