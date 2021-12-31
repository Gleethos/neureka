/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix;

import neureka.backend.standard.operations.linear.fast.ProgrammingError;
import neureka.backend.standard.operations.linear.fast.matrix.store.Core;
import neureka.backend.standard.operations.linear.fast.matrix.store.MatrixCore;
import neureka.backend.standard.operations.linear.fast.matrix.store.MatrixCore.Factory;
import neureka.backend.standard.operations.linear.fast.matrix.store.TransformableRegion;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.structure.Access2D;

/**
 * A base class for, easy to use, immutable (thread safe) matrices with a rich feature set. This class handles
 * a lot of complexity, and makes choices, for you.
 *
 */
public abstract
        class BasicMatrix<N extends Comparable<N>, M extends BasicMatrix<N, M>>
        implements
            Matrix2D<N, M>,
            Access2D.Collectable<N, Core<N>>
{

    private final MatrixCore<N> _core;
    private final Factory<N, ?> _factory;

    BasicMatrix(final MatrixCore<N> store) {

        super();

        _core = store;
        _factory = store.factory();
    }

    @Override
    public int size() {
        return _core.size();
    }

    @Override
    public int numberOfColumns() {
        return _core.numberOfColumns();
    }

    @Override
    public int numberOfRows() {
        return _core.numberOfRows();
    }

    @Override
    public double doubleValue(final long index) {
        return _core.doubleValue(index);
    }

    @Override
    public double doubleValue(final long i, final long j) {
        return _core.doubleValue(i, j);
    }

    @Override
    public N get(final long index) {
        return _core.get(index);
    }

    @Override
    public N getAt(final long aRow, final long aColumn) {
        return _core.getAt(aRow, aColumn);
    }

    @Override
    public M multiply(final M multiplicand, Object otherData) {

        ProgrammingError.throwIfMultiplicationNotPossible(_core, multiplicand);

        Access2D.Collectable<N, TransformableRegion<N>> internals = this.toInternals(multiplicand);

        MatrixCore<N> retVal = _factory.make(internals.numberOfRows(), internals.numberOfColumns(), null);

        internals.supplyToTrans((TransformableRegion<N>) retVal);

        return this.getResultFactory()
                    .instantiate(
                        _core.multiply(
                            retVal,
                            otherData
                        )
                    );
    }

    @Override
    public final void supplyToTrans(final Core<N> receiver) {
        _core.supplyToTrans(receiver);
    }

    @Override
    public final String toString() {
        return Access2D.toString(this);
    }

    abstract Access2D.Collectable<N, TransformableRegion<N>> toInternals(Access1D<?> matrix);

    abstract MatrixFactory<N, M> getResultFactory();

    final MatrixCore<N> getStore() {
        return _core;
    }

}
