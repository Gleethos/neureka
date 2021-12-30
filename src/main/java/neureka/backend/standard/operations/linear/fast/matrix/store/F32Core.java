/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.store;

import neureka.backend.standard.operations.linear.fast.ProgrammingError;
import neureka.backend.standard.operations.linear.fast.array.Array2D;
import neureka.backend.standard.operations.linear.fast.array.DenseArray;
import neureka.backend.standard.operations.linear.fast.array.F32Array;
import neureka.backend.standard.operations.linear.fast.array.operation.FillMatchingSingle;
import neureka.backend.standard.operations.linear.fast.concurrent.WorkScheduler;
import neureka.backend.standard.operations.linear.fast.function.BinaryFunction;
import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;
import neureka.backend.standard.operations.linear.fast.matrix.operation.MultiplyBoth;
import neureka.backend.standard.operations.linear.fast.matrix.operation.MultiplyLeft;
import neureka.backend.standard.operations.linear.fast.matrix.operation.MultiplyNeither;
import neureka.backend.standard.operations.linear.fast.matrix.operation.MultiplyRight;
import neureka.backend.standard.operations.linear.fast.structure.*;


/**
 * A {@linkplain float} implementation of {@linkplain Core}.
 *
 */
public final class F32Core extends F32Array implements Core<Double> {

    public static final MatrixCore.Factory<Double, F32Core> FACTORY = new PrimitiveFactory<F32Core>() {

        @Override
        public DenseArray.Factory<Double> array() {
            return F32Array.FACTORY;
        }

        public F32Core copy(final Access2D<?> source) {

            final int tmpRowDim = (int) source.numberOfRows();
            final int tmpColDim = (int) source.numberOfColumns();

            final F32Core retVal = new F32Core(tmpRowDim, tmpColDim);

            if (tmpColDim > FillMatchingSingle.THRESHOLD) {
                final WorkScheduler localScheduler = new WorkScheduler() {

                    @Override
                    public void work(final int aFirst, final int aLimit) {
                        FillMatchingSingle.copy(retVal.data, tmpRowDim, aFirst, aLimit, source);
                    }

                };
                localScheduler.invoke(0, tmpColDim, FillMatchingSingle.THRESHOLD);
            }
            else
                FillMatchingSingle.copy(retVal.data, tmpRowDim, 0, tmpColDim, source);

            return retVal;
        }

        public F32Core make(final long rows, final long columns, Object data) {
            return ( data == null
                        ? new F32Core((int) rows, (int) columns)
                        : new F32Core((int) rows, (int) columns, (float[]) data)
                    );
        }

    };

    static F32Core cast(
            final Access1D<Double> matrix
    ) {
        if ( matrix instanceof F32Core )
            return (F32Core) matrix;
        else
            throw new IllegalArgumentException();
    }

    private final MultiplyBoth.Primitive      _multiplyBoth;
    private final MultiplyLeft.Primitive32    _multiplyLeft;
    private final MultiplyNeither.Primitive32 _multiplyNeither;
    private final MultiplyRight.Primitive32   _multiplyRight;
    private final int _colDim;
    private final int _rowDim;
    private final Array2D<Double> _utility;

    F32Core(final int numbRows, final int numbCols, final float[] dataArray) {

        super(dataArray);

        _rowDim = numbRows;
        _colDim = numbCols;

        _utility = this.wrapInArray2D(_rowDim);

        _multiplyBoth = MultiplyBoth.newPrimitive32(_rowDim, _colDim);
        _multiplyLeft = MultiplyLeft.newPrimitive32(_rowDim, _colDim);
        _multiplyRight = MultiplyRight.newPrimitive32(_rowDim, _colDim);
        _multiplyNeither = MultiplyNeither.newPrimitive32(_rowDim, _colDim);
    }

    F32Core(final long numbRows, final long numbCols) {

        super(Math.toIntExact(numbRows * numbCols));

        _rowDim = Math.toIntExact(numbRows);
        _colDim = Math.toIntExact(numbCols);

        _utility = this.wrapInArray2D(_rowDim);

        _multiplyBoth = MultiplyBoth.newPrimitive32(_rowDim, _colDim);
        _multiplyLeft = MultiplyLeft.newPrimitive32(_rowDim, _colDim);
        _multiplyRight = MultiplyRight.newPrimitive32(_rowDim, _colDim);
        _multiplyNeither = MultiplyNeither.newPrimitive32(_rowDim, _colDim);
    }

    @Override
    public int numberOfColumns() {
        return _colDim;
    }

    @Override
    public int numberOfRows() {
        return _rowDim;
    }

    @Override
    public double doubleValue(final long row, final long col) {
        return _utility.doubleValue(row, col);
    }

    public void fillByMultiplying(final Access1D<Double> left, final Access1D<Double> right) {

        int complexity = Math.toIntExact(left.size() / this.numberOfRows());
        if (complexity != Math.toIntExact(right.size() / this.numberOfColumns())) {
            ProgrammingError.throwForMultiplicationNotPossible();
        }
        if ( left instanceof F32Core ) {
            if ( right instanceof F32Core )
                _multiplyNeither.invoke(data, F32Core.cast(left).data, complexity, F32Core.cast(right).data);
            else
                _multiplyRight.invoke(data, F32Core.cast(left).data, complexity, right);
        }
        else if ( right instanceof F32Core )
            _multiplyLeft.invoke(data, left, complexity, F32Core.cast(right).data);
        else
            _multiplyBoth.invoke(this, left, complexity, right);
    }

    public float floatValue(final long row, final long col) {
        return _utility.floatValue(row, col);
    }

    public Double getAt(final long rowIndex, final long colIndex) {
        return _utility.getAt(rowIndex, colIndex);
    }

    public int getColDim() {
        return _colDim;
    }

    public int getRowDim() {
        return _rowDim;
    }

    @Override
    public void operateAll(final UnaryFunction<Double> modifier) {
        this.modify(0, _rowDim * _colDim, 1, modifier);
    }

    @Override
    public void operateWith(final Access1D<Double> left, final BinaryFunction<Double> function) {
        _utility.operateWith(left, function);
    }

    @Override
    public void operate(final BinaryFunction<Double> function, final Access1D<Double> right) {
        _utility.operate(function, right);
    }

    public MatrixCore<Double> multiply(final MatrixCore<Double> right, Object otherData) {
        F32Core retVal = FACTORY.make(_rowDim, right.numberOfColumns(), otherData);
        if ( right instanceof F32Core )
            retVal._multiplyNeither.invoke(retVal.data, data, _colDim, F32Core.cast(right).data);
        else
            retVal._multiplyRight.invoke(retVal.data, data, _colDim, right);
        return retVal;
    }

    public MatrixCore.Factory<Double, ?> factory() {
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
