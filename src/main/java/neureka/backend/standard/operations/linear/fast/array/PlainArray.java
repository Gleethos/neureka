/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.function.BinaryFunction;
import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;

import java.util.RandomAccess;

/**
 * Typically a primitive Java array like <code>double[]</code>.
 */
abstract class PlainArray<N extends Comparable<N>> extends DenseArray<N> implements RandomAccess {

    PlainArray(final Factory<N> factory, final int size) {

        super(factory);

        if (size > MAX_ARRAY_SIZE) {
            throw new IllegalArgumentException("Array too large!");
        }
    }

    @Override
    public final double doubleValue(final long index) {
        // No Math.toIntExact() here, be as direct as possible
        return this.doubleValue((int) index);
    }

    @Override
    public final void fillOneAccess(final long index, final Access1D<?> values, final long valueIndex) {
        //this.fillOne(Math.toIntExact(index), values, valueIndex);
    }

    @Override
    public final float floatValue(final long index) {
        // No Math.toIntExact() here, be as direct as possible
        return this.floatValue((int) index);
    }

    @Override
    public final N get(final long index) {
        // No Math.toIntExact() here, be as direct as possible
        return this.get((int) index);
    }

    @Override
    public final void set(final long index, final double value) {
        // No Math.toIntExact() here, be as direct as possible
        this.set((int) index, value);
    }

    protected abstract void add(int index, Comparable<?> addend);

    protected abstract void add(int index, double addend);

    protected abstract void add(int index, float addend);

    protected abstract double doubleValue(int index);

    protected abstract float floatValue(final int index);

    protected abstract N get(final int index);

    protected abstract void modify(int first, int limit, int step, Access1D<N> left, BinaryFunction<N> function);

    protected abstract void modify(int first, int limit, int step, BinaryFunction<N> function, Access1D<N> right);

    protected abstract void modify(int first, int limit, int step, UnaryFunction<N> function);

    @Override
    protected final void modify(final long first, final long limit, final long step, final Access1D<N> left, final BinaryFunction<N> function) {
        this.modify(Math.toIntExact(first), Math.toIntExact(limit), Math.toIntExact(step), left, function);
    }

    @Override
    protected final void modify(final long first, final long limit, final long step, final BinaryFunction<N> function, final Access1D<N> right) {
        this.modify(Math.toIntExact(first), Math.toIntExact(limit), Math.toIntExact(step), function, right);
    }

    @Override
    protected final void modify(final long first, final long limit, final long step, final UnaryFunction<N> function) {
        this.modify(Math.toIntExact(first), Math.toIntExact(limit), Math.toIntExact(step), function);
    }

    protected abstract void set(final int index, final Comparable<?> number);

    protected abstract void set(final int index, final double value);

    protected abstract void set(final int index, final float value);

}
