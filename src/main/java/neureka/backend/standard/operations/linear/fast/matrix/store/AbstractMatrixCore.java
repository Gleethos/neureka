/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.structure.Access2D;

abstract class AbstractMatrixCore<N extends Comparable<N>> implements MatrixCore<N> {

    private final int _colDim;
    private final int _rowDim;

    protected AbstractMatrixCore(final int numberOfRows, final int numberOfColumns) {
        super();
        _rowDim = numberOfRows;
        _colDim = numberOfColumns;
    }

    public int numberOfColumns() {
        return _colDim;
    }

    public int numberOfRows() {
        return _rowDim;
    }

    public final int getColDim() {
        return _colDim;
    }

    public final int getRowDim() {
        return _rowDim;
    }

    public int limitOfColumn(final int col) {
        return _rowDim;
    }

    public int limitOfRow(final int row) {
        return _colDim;
    }

    @Override
    public final String toString() {
        return Access2D.toString(this);
    }

}
