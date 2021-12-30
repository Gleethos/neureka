/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.function.BinaryFunction;
import neureka.backend.standard.operations.linear.fast.function.FunctionSet;
import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.structure.Mutate1D;
import neureka.backend.standard.operations.linear.fast.structure.Structure1D;

/**
 * <p>
 * A BasicArray is 1-dimensional, but designed to easily be extended or encapsulated, and then treated as
 * arbitrary-dimensional. It stores/handles (any subclass of) {@linkplain Comparable} elements
 * depending on the subclass/implementation.
 * </p>
 * <p>
 * This abstract class defines a set of methods to access and modify array elements. It does not "know"
 * anything about linear algebra or similar.
 * </p>
 *
 */
public abstract
        class BasicArray<N extends Comparable<N>>
        implements
            Access1D<N>,
            Mutate1D,
            Mutate1D.Fillable<N>,
            Mutate1D.Modifiable<N>,
            Structure1D
{

    public static final class Factory<N extends Comparable<N>> extends ArrayFactory<N, BasicArray<N>> {

        private final DenseArray.Factory<N> myDenseFactory;

        Factory(final DenseArray.Factory<N> denseFactory) {
            super();
            myDenseFactory = denseFactory;
        }

        @Override
        public FunctionSet<N> forFunctions() {
            return myDenseFactory.forFunctions();
        }

    }

    private final ArrayFactory<N, ?> myFactory;

    @SuppressWarnings("unused")
    private BasicArray() {
        this(null);
    }

    protected BasicArray(final ArrayFactory<N, ?> factory) {
        super();
        myFactory = factory;
    }

    @Override
    public boolean equals(final Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof BasicArray)) {
            return false;
        }
        BasicArray other = (BasicArray) obj;
        if (myFactory == null) {
            if (other.myFactory != null) {
                return false;
            }
        } else if (!myFactory.equals(other.myFactory)) {
            return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + (myFactory == null ? 0 : myFactory.hashCode());
        return result;
    }

    public void operateAll(final UnaryFunction<N> modifier) {
        this.modify(0L, this.size(), 1L, modifier);
    }

    public void operateWith(final Access1D<N> left, final BinaryFunction<N> function) {
        long limit = Math.min(left.size(), this.size());
        this.modify(0L, limit, 1L, left, function);
    }

    public void operate(final BinaryFunction<N> function, final Access1D<N> right) {
        long limit = Math.min(this.size(), right.size());
        this.modify(0L, limit, 1L, function, right);
    }

    @Override
    public String toString() {
        return Access1D.toString(this);
    }

    protected abstract void modify(long first, long limit, long step, Access1D<N> left, BinaryFunction<N> function);

    protected abstract void modify(long first, long limit, long step, BinaryFunction<N> function, Access1D<N> right);

    protected abstract void modify(long first, long limit, long step, UnaryFunction<N> function);

    /**
     * A utility facade that conveniently/consistently presents the {@linkplain BasicArray}
     * as a one-dimensional array. Note that you will modify the actual array by accessing it through this
     * facade.
     */
    protected final Array1D<N> wrapInArray1D() {
        return new Array1D<>(this);
    }

    /**
     * A utility facade that conveniently/consistently presents the {@linkplain BasicArray}
     * as a two-dimensional array. Note that you will modify the actual array by accessing it through this
     * facade.
     */
    protected final Array2D<N> wrapInArray2D(final int structure) {
        return new Array2D<>(this, structure);
    }

}
