package neureka.backend.main.operations.linear.internal;

import neureka.Tsr;
import neureka.backend.main.operations.linear.internal.blas.MatMul;

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

    public M64 multiply( boolean rowMajor, M64 matrix, double[] otherData ) {

        int otherColCount = matrix._colCount;

        M64 retVal = new M64(_rowCount, otherColCount, otherData);

        MatMul
            .operationForF64( rowMajor, _rowCount, otherColCount )
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

}
