/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix;

import neureka.backend.standard.operations.linear.fast.matrix.store.*;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.structure.Access2D;

/**
 * A matrix (linear algebra) with 32-bit primitive float elements.
 *
 * @see BasicMatrix
 */
public final class MatrixF32 extends BasicMatrix<Double, MatrixF32> {

    public static final class Factory extends
            MatrixFactory<Double, MatrixF32> {

        Factory() {
            super(MatrixF32.class, F32Core.FACTORY);
        }

        @Override
        public MatrixF32 make(long rows, long columns, Object data) {
            return makeFilled(rows, columns, data);
        }
    }

    public static final Factory FACTORY = new Factory();

    /**
     * This method is for internal use only - YOU should NOT use it!
     */
    MatrixF32(final MatrixCore<Double> aStore) {
        super(aStore);
    }

    @SuppressWarnings("unchecked")
    @Override
    Access2D.Collectable<Double, TransformableRegion<Double>> toInternals(final Access1D<?> matrix) {
        if (matrix instanceof MatrixF32) {
            return ((MatrixF32) matrix).getStore();
        }
        else
            throw new IllegalArgumentException();
    }

    @Override
    Factory getResultFactory() {
        return FACTORY;
    }

    public float[] getData() {
        return ((F32Core)getStore()).data;
    }

}
