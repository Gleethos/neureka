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
import neureka.backend.standard.operations.linear.fast.matrix.operation.Multiply;
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

    private final Multiply.Primitive32 _multiplyNeither;
    private final int _colDim;
    private final int _rowDim;
    private final Array2D<Double> _utility;

    F32Core(final int numbRows, final int numbCols, final float[] dataArray) {

        super(dataArray);

        _rowDim = numbRows;
        _colDim = numbCols;

        _utility = this.wrapInArray2D(_rowDim);
        _multiplyNeither = Multiply.newPrimitive32(_rowDim, _colDim);
    }

    F32Core(final long numbRows, final long numbCols) {

        super(Math.toIntExact(numbRows * numbCols));

        _rowDim = Math.toIntExact(numbRows);
        _colDim = Math.toIntExact(numbCols);

        _utility = this.wrapInArray2D(_rowDim);
        _multiplyNeither = Multiply.newPrimitive32(_rowDim, _colDim);
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
        _multiplyNeither.invoke(data, F32Core.cast(left).data, complexity, F32Core.cast(right).data);
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

    public MatrixCore<Double> multiply(final MatrixCore<Double> right, Object otherData) {
        F32Core retVal = FACTORY.make(_rowDim, right.numberOfColumns(), otherData);
        retVal._multiplyNeither.invoke(retVal.data, data, _colDim, F32Core.cast(right).data);
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
