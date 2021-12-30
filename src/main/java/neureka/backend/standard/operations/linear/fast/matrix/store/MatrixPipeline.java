/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.function.BinaryFunction;
import neureka.backend.standard.operations.linear.fast.structure.Access2D;

abstract class MatrixPipeline<N extends Comparable<N>> implements Access2D.Collectable<N, TransformableRegion<N>> {

    static final class BinaryOperatorRight<N extends Comparable<N>> extends MatrixPipeline<N> {

        private final BinaryFunction<N> myOperator;
        private final Access2D<N> myRight;

        BinaryOperatorRight(final Access2D.Collectable<N, TransformableRegion<N>> left, final BinaryFunction<N> operator, final Access2D<N> right) {
            super(left);
            myRight = right;
            myOperator = operator;
        }

        @Override
        public void supplyToTrans(final TransformableRegion<N> receiver) {
            this.getContext().supplyToTrans(receiver);
            receiver.operate(myOperator, myRight);
        }

    }

    private final int _columnsCount;
    private final Access2D.Collectable<N, TransformableRegion<N>> _context;
    private final int _rowsCount;

    MatrixPipeline(final Access2D.Collectable<N, TransformableRegion<N>> context) {
        this(context, context.numberOfRows(), context.numberOfColumns());
    }

    MatrixPipeline(final Access2D.Collectable<N, TransformableRegion<N>> context, final long rowsCount, final long columnsCount) {
        super();
        _context = context;
        _rowsCount = (int) rowsCount;
        _columnsCount = (int) columnsCount;
    }

    public final int numberOfColumns() {
        return _columnsCount;
    }

    public final int numberOfRows() {
        return _rowsCount;
    }

    @Override
    public final String toString() {
        return _rowsCount + "x" + _columnsCount + " " + this.getClass();
    }

    final Access2D.Collectable<N, TransformableRegion<N>> getContext() {
        return _context;
    }

}
