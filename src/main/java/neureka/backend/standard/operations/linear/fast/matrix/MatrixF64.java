/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix;

import neureka.backend.standard.operations.linear.fast.matrix.store.*;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.structure.Access2D;

/**
 * A matrix (linear algebra) with 64-bit primitive double elements.
 *
 * @see BasicMatrix
 */
public final class MatrixF64 extends BasicMatrix<Double, MatrixF64> {

    public static final class Factory extends
            MatrixFactory<
                Double,
                MatrixF64
            >
    {

        Factory() {
            super(MatrixF64.class, F64Core.FACTORY);
        }

        @Override
        public MatrixF64 make(long rows, long columns, Object data) {
            return makeFilled(rows, columns, data);
        }
    }

    public static final Factory FACTORY = new Factory();

    /**
     * This method is for internal use only - YOU should NOT use it!
     */
    MatrixF64(final MatrixCore<Double> aStore) {
        super(aStore);
    }

    @SuppressWarnings("unchecked")
    @Override
    Access2D.Collectable<Double, TransformableRegion<Double>> toInternals(final Access1D<?> matrix) {

        if (matrix instanceof MatrixF64) {
            return ((MatrixF64) matrix).getStore();
        }
        else
            throw new IllegalArgumentException();
    }

    @Override
    Factory getResultFactory() {
        return FACTORY;
    }

    public double[] getData() {
        return ((F64Core)getStore()).data;
    }

}
