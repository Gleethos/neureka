package neureka.backend.standard.operations.linear.internal;

import neureka.backend.standard.operations.linear.internal.operation.matrix.Vectorizable;

public class M64 {

    private final double[] _data;
    private final int _rowCount, _colCount;

    public M64(
            int rowCount,
            int colCount,
            double[] data
    ) {
        _data = data;
        _rowCount = rowCount; // M
        _colCount = colCount; // N
    }

    public M64 multiply( M64 matrix, double[] otherData ) {

        int otherColCount = matrix._colCount;

        M64 retVal = new M64(_rowCount, otherColCount, otherData);

        Vectorizable
            .newVOF64(_rowCount, otherColCount)
            .invoke(
                retVal._data, _data, _colCount, matrix._data
            );
            return retVal;
    }

    public int getRowDim() {return _rowCount;}

    public int getColDim() {return _colCount;}

    public double[] getData() {
        return _data;
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName()+"["+_rowCount+"x"+_colCount+"]";
    }

}
