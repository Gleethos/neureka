package neureka.backend.main.operations.linear.internal;

import neureka.Tsr;
import neureka.backend.main.operations.linear.internal.blas.MatMul;

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

    public M32 multiply( boolean rowMajor, M32 matrix, float[] otherData ) {

        int otherColCount = matrix._colCount;

        M32 retVal = new M32(_rowCount, otherColCount, otherData);

        MatMul
                .operationForF32( rowMajor, _rowCount, otherColCount )
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
        return this.getClass().getSimpleName()+"["+_rowCount+"x"+_colCount+"]"
                + Tsr.of(
                            Float.class,
                            new int[]{_rowCount,_colCount},
                            _data
                        )
                        .toString( it ->
                                    it.setHasShape(false)
                                      .setHasSlimNumbers(true)
                                      .setHasValue(true)
                                      .setIsMultiline(true)
                                      .setIsLegacy(true)
                                      .setIsScientific(true)
                                      .setIsCellBound(true)
                                      .setCellSize(4)
                                );
    }

}
