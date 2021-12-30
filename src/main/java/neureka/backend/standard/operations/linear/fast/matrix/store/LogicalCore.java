/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

/**
 * Logical stores are (intended to be) immutable.
 *
 */
abstract class LogicalCore<N extends Comparable<N>> extends AbstractMatrixCore<N> {

    private final MatrixCore<N> _base;

    protected LogicalCore(final MatrixCore<N> base, final int rowsCount, final int columnsCount) {

        super(rowsCount, columnsCount);

        _base = base;
    }

    protected LogicalCore(final MatrixCore<N> base, final long rowsCount, final long columnsCount) {
        this(base, Math.toIntExact(rowsCount), Math.toIntExact(columnsCount));
    }

    public Factory<N, ?> factory() {
        return _base.factory();
    }

    final MatrixCore<N> base() {
        return _base;
    }

}
