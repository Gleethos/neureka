package neureka.backend.standard.operations.linear.internal;

import neureka.backend.standard.operations.linear.internal.operation.matrix.Vectorizable;

public class M32 {

    private final float[] _data;
    private final int _rowCount, _colCount;

    public M32(
            int rowCount,
            int colCount,
            float[] data
    ) {
        _data = data;
        _rowCount = rowCount; // M
        _colCount = colCount; // N
    }

    public M32 multiply( M32 matrix, float[] otherData ) {

        int otherColCount = matrix._colCount;

        M32 retVal = new M32(_rowCount, otherColCount, otherData);

        Vectorizable
                .newVOF32(_rowCount, otherColCount)
                .invoke(
                        retVal._data, _data, _colCount, matrix._data
                );
        return retVal;
    }

    public int getRowDim() {return _rowCount;}

    public int getColDim() {return _colCount;}

    public float[] getData() {
        return _data;
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName()+"["+_rowCount+"x"+_colCount+"]";
    }

}
