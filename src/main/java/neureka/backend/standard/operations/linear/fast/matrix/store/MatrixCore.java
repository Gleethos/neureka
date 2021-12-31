/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.function.FunctionSet;
import neureka.backend.standard.operations.linear.fast.matrix.Matrix2D;
import neureka.backend.standard.operations.linear.fast.structure.*;
import neureka.backend.standard.operations.linear.fast.type.NumberDefinition;

public interface MatrixCore<N extends Comparable<N>>
        extends
            Matrix2D<N, MatrixCore<N>>,
            Access2D.Collectable<N, TransformableRegion<N>>,
            Structure2D,
        Size
{
    interface Factory<N extends Comparable<N>, I extends Core<N>> extends Factory2D.Dense<I> {

        FunctionSet<N> forFunctions();

    }

    default double doubleValue(final long row, final long col) {
        return NumberDefinition.doubleValue(this.getAt(row, col));
    }

    default void multiplyAccess(final Access1D<N> right, final TransformableRegion<N> target) {
        target.fillByMultiplying(this, right);
    }

    default MatrixCore<N> multiply(final MatrixCore<N> right, Object otherData) {

        long tmpCountRows = this.numberOfRows();
        long tmpCountColumns = right.numberOfColumns();

        Core<N> retVal = this.factory().make(tmpCountRows, tmpCountColumns, otherData);

        this.multiplyAccess(right, retVal);

        return retVal;
    }

    Factory<N, ?> factory();

    default void supplyToTrans(final TransformableRegion<N> receiver) {
        receiver.fillMatchingAccess(this);
    }

}
