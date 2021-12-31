/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix;

import neureka.backend.standard.operations.linear.fast.algebra.Operation;
import neureka.backend.standard.operations.linear.fast.algebra.ScalarOperation;
import neureka.backend.standard.operations.linear.fast.matrix.store.MatrixCore;
import neureka.backend.standard.operations.linear.fast.structure.Access2D;
import neureka.backend.standard.operations.linear.fast.structure.Size;
import neureka.backend.standard.operations.linear.fast.structure.Structure2D;

/**
 * Definition of what's common to {@link BasicMatrix} and {@link MatrixCore}. At this point, at least, it is
 * not recommended to write any code in terms of this interface. It's new, the definition may change and it
 * may even be removed again.
 *
 */
public interface Matrix2D<N extends Comparable<N>, M extends Matrix2D<N, M>>
        extends
        Access2D<N>,
        Structure2D,
        Size,
        Operation.Multiplication<M>,
        ScalarOperation.Multiplication<M, N>
{

}
