/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.ProgrammingError;
import neureka.backend.standard.operations.linear.fast.array.Array2D;
import neureka.backend.standard.operations.linear.fast.array.F64Array;
import neureka.backend.standard.operations.linear.fast.array.operation.*;
import neureka.backend.standard.operations.linear.fast.concurrent.WorkScheduler;
import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;
import neureka.backend.standard.operations.linear.fast.matrix.operation.MultiplyBoth;
import neureka.backend.standard.operations.linear.fast.matrix.operation.MultiplyLeft;
import neureka.backend.standard.operations.linear.fast.matrix.operation.MultiplyNeither;
import neureka.backend.standard.operations.linear.fast.matrix.operation.MultiplyRight;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.structure.Access2D;

public final class F64Core extends F64Array implements Core<Double> {

    public static final MatrixCore.Factory<Double, F64Core> FACTORY = new PrimitiveFactory<F64Core>() {

        public F64Core copy(final Access2D<?> source) {

            final int tmpRowDim = source.numberOfRows();
            final int tmpColDim = source.numberOfColumns();

            final F64Core retVal = new F64Core(tmpRowDim, tmpColDim);

            if (tmpColDim > FillMatchingSingle.THRESHOLD) {
                final WorkScheduler tmpConquerer = new WorkScheduler() {

                    @Override
                    public void work(final int aFirst, final int aLimit) {
                        FillMatchingSingle.copy(retVal.data, tmpRowDim, aFirst, aLimit, source);
                    }

                };
                tmpConquerer.invoke(0, tmpColDim, FillMatchingSingle.THRESHOLD);
            }
            else
                FillMatchingSingle.copy(retVal.data, tmpRowDim, 0, tmpColDim, source);

            return retVal;
        }
        @Override
        public F64Core make(final long rows, final long columns, Object data) {
            return ( data == null
                        ? new F64Core((int) rows, (int) columns)
                        : new F64Core((int) rows, (int) columns, (double[]) data)
                    );
        }

    };


    static F64Core cast(final Access1D<Double> matrix) {
        if (matrix instanceof F64Core) {
            return (F64Core) matrix;
        }
        else
            throw new IllegalArgumentException();
    }

    private final MultiplyBoth.Primitive _multiplyBoth;
    private final MultiplyLeft.Primitive64 _multiplyLeft;
    private final MultiplyNeither.Primitive64 _multiplyNeither;
    private final MultiplyRight.Primitive64 _multiplyRight;
    private final int _colDim;
    private final int _rowDim;
    private final Array2D<Double> _utility;

    F64Core(final int numbRows, final int numbCols, final double[] dataArray) {

        super(dataArray);

        _rowDim = numbRows;
        _colDim = numbCols;

        _utility = this.wrapInArray2D(_rowDim);

        _multiplyBoth = MultiplyBoth.newPrimitive64(_rowDim, _colDim);
        _multiplyLeft = MultiplyLeft.newPrimitive64(_rowDim, _colDim);
        _multiplyRight = MultiplyRight.newPrimitive64(_rowDim, _colDim);
        _multiplyNeither = MultiplyNeither.newPrimitive64(_rowDim, _colDim);
    }

    F64Core(final long numbRows, final long numbCols) {

        super(Math.toIntExact(numbRows * numbCols));

        _rowDim = Math.toIntExact(numbRows);
        _colDim = Math.toIntExact(numbCols);

        _utility = this.wrapInArray2D(_rowDim);

        _multiplyBoth = MultiplyBoth.newPrimitive64(_rowDim, _colDim);
        _multiplyLeft = MultiplyLeft.newPrimitive64(_rowDim, _colDim);
        _multiplyRight = MultiplyRight.newPrimitive64(_rowDim, _colDim);
        _multiplyNeither = MultiplyNeither.newPrimitive64(_rowDim, _colDim);
    }

    public int numberOfColumns() {
        return _colDim;
    }

    public int numberOfRows() {
        return _rowDim;
    }

    public double doubleValue(final long row, final long col) {
        return _utility.doubleValue(row, col);
    }

    public void fillByMultiplying(final Access1D<Double> left, final Access1D<Double> right) {

        final int complexity = Math.toIntExact(left.size() / this.numberOfRows());
        if (complexity != Math.toIntExact(right.size() / this.numberOfColumns())) {
            ProgrammingError.throwForMultiplicationNotPossible();
        }

        if ( left instanceof F64Core ) {
            if ( right instanceof F64Core )
                _multiplyNeither.invoke(data, F64Core.cast(left).data, complexity, F64Core.cast(right).data);
            else
                _multiplyRight.invoke(data, F64Core.cast(left).data, complexity, right);
        }
        else if ( right instanceof F64Core )
            _multiplyLeft.invoke(data, left, complexity, F64Core.cast(right).data);
        else
            _multiplyBoth.invoke(this, left, complexity, right);
    }

    @Override
    public void fillMatchingAccess(final Access1D<?> values) {
        super.fillMatchingAccess(values);
    }

    @Override
    public Double getAt(final long row, final long col) {
        return _utility.getAt(row, col);
    }

    @Override
    public int getColDim() {
        return _colDim;
    }

    @Override
    public int getRowDim() {
        return _rowDim;
    }

    @Override
    public void operateAll(final UnaryFunction<Double> modifier) {

        if ( _colDim > ModifyAll.THRESHOLD ) {

            final WorkScheduler scheduler = new WorkScheduler() {
                @Override
                public void work(final int first, final int limit) {
                    F64Core.this.modify(_rowDim * first, _rowDim * limit, 1, modifier);
                }

            };
            scheduler.invoke(0, _colDim, ModifyAll.THRESHOLD);
        }
        else
            this.modify(0, _rowDim * _colDim, 1, modifier);
    }

    public MatrixCore<Double> multiply(final MatrixCore<Double> right, Object otherData) {

        F64Core retVal = FACTORY.make(_rowDim, right.numberOfColumns(), otherData);

        if (right instanceof F64Core)
            retVal._multiplyNeither.invoke(retVal.data, data, _colDim, F64Core.cast(right).data);
        else
            retVal._multiplyRight.invoke(retVal.data, data, _colDim, right);

        return retVal;
    }

    public MatrixCore.Factory<Double, F64Core> factory() {
        return FACTORY;
    }


    public void set(final long row, final long col, final double value) {
        _utility.set(row, col, value);
    }

    @Override
    public String toString() {
        return Access2D.toString(this);
    }

}
