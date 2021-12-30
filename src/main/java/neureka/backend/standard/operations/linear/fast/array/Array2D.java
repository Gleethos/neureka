/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.function.*;
import neureka.backend.standard.operations.linear.fast.structure.*;

/**
 * Array2D
 *
 */
public final class Array2D<N extends Comparable<N>>
        implements
            Structure2D,
            Structure1D,
            Mutate1D.Modifiable<N>,
            Mutate2D.Receiver<N>,
            Access2D<N>
{

    private final int _columnsCount;
    private final BasicArray<N> _delegate;
    private final int _rowsCount;

    Array2D(
            final BasicArray<N> delegate,
            final int structure
    ) {
        super();
        _delegate = delegate;
        _rowsCount = (int) structure;
        _columnsCount = structure == 0 ? 0 : (int) (delegate.size() / structure);
    }

    @Override
    public int size() {
        return _delegate.size();
    }

    @Override
    public int numberOfColumns() {
        return _columnsCount;
    }

    @Override
    public int numberOfRows() {
        return _rowsCount;
    }

    @Override
    public double doubleValue(final long index) {
        return _delegate.doubleValue(index);
    }

    @Override
    public double doubleValue(final long row, final long col) {
        return _delegate.doubleValue(Structure2D.index(_rowsCount, row, col));
    }

    @Override
    public void fillOneAccess(final long index, final Access1D<?> values, final long valueIndex) {
        _delegate.fillOneAccess(index, values, valueIndex);
    }

    @Override
    public N get(final long index) {
        return _delegate.get(index);
    }

    @Override
    public N getAt(final long row, final long col) {
        return _delegate.get(Structure2D.index(_rowsCount, row, col));
    }

    @Override
    public void operateAll(final UnaryFunction<N> modifier) {
        _delegate.modify(0L, this.size(), 1L, modifier);
    }

    @Override
    public void operateWith(final Access1D<N> left, final BinaryFunction<N> function) {
        _delegate.modify(0L, this.size(), 1L, left, function);
    }

    @Override
    public void operate(final BinaryFunction<N> function, final Access1D<N> right) {
        _delegate.modify(0L, this.size(), 1L, function, right);
    }

    @Override
    public void set(final long index, final double value) {
        _delegate.set(index, value);
    }

    @Override
    public void set(final long row, final long col, final double value) {
        _delegate.set(Structure2D.index(_rowsCount, row, col), value);
    }

    @Override
    public String toString() {
        return Access2D.toString(this);
    }

}
