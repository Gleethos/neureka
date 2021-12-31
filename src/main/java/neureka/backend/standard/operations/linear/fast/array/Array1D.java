/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.function.BinaryFunction;
import neureka.backend.standard.operations.linear.fast.function.FunctionSet;
import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.structure.FactorySupplement;
import neureka.backend.standard.operations.linear.fast.structure.Mutate1D;
import neureka.backend.standard.operations.linear.fast.structure.Size;

import java.util.RandomAccess;

/**
 * Array1D
 *
 */
public final
        class Array1D<N extends Comparable<N>>
        implements
            Size,
            Mutate1D,
            Mutate1D.Fillable<N>,
            Access1D<N>,
            RandomAccess
{

    public static final class Factory<N extends Comparable<N>>
            implements
            FactorySupplement
    {

        private final BasicArray.Factory<N> _delegate;

        Factory(final DenseArray.Factory<N> denseArray) {
            super();
            _delegate = new BasicArray.Factory<>(denseArray);
        }

        @Override
        public FunctionSet<N> forFunctions() {
            return _delegate.forFunctions();
        }

    }

    public final long length;

    private final BasicArray<N> _delegate;
    private final long myFirst;
    private final long myLimit;
    private final long myStep;

    Array1D(final BasicArray<N> delegate) {
        this(delegate, 0L, delegate.size(), 1L);
    }

    Array1D(final BasicArray<N> delegate, final long first, final long limit, final long step) {

        super();

        _delegate = delegate;

        myFirst = first;
        myLimit = limit;
        myStep = step;

        length = (myLimit - myFirst) / myStep;
    }

    @Override
    public double doubleValue(final long index) {
        return _delegate.doubleValue(this.convert(index));
    }

    @Override
    public void fillOneAccess(final long index, final Access1D<?> values, final long valueIndex) {
        _delegate.fillOneAccess(this.convert(index), values, valueIndex);
    }

    @Override
    public N get(final long index) {
        return _delegate.get(this.convert(index));
    }

    @Override
    public void set(final long index, final double value) {
        _delegate.set(this.convert(index), value);
    }

    @Override
    public int size() {
        return Math.toIntExact(length);
    }

    @Override
    public String toString() {
        return Access1D.toString(this);
    }

    /**
     * Convert an external (public API) index to the corresponding internal
     */
    private long convert(final long index) {
        return myFirst + myStep * index;
    }

}
