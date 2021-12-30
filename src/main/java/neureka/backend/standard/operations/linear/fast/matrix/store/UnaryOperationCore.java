/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;

final class UnaryOperationCore<N extends Comparable<N>> extends LogicalCore<N> {

    private final UnaryFunction<N> _operator;

    UnaryOperationCore(final MatrixCore<N> base, final UnaryFunction<N> operator) {

        super(base, base.numberOfRows(), base.numberOfColumns());

        _operator = operator;
    }

    public double doubleValue(final long row, final long col) {
        return _operator.invoke(this.base().doubleValue(row, col));
    }

    public N getAt(final long row, final long col) {
        return _operator.invoke(this.base().getAt(row, col));
    }

}
