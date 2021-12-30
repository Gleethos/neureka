/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

import neureka.backend.standard.operations.linear.fast.array.operation.COPY;
import neureka.backend.standard.operations.linear.fast.array.operation.FillMatchingSingle;
import neureka.backend.standard.operations.linear.fast.array.operation.OperationBinary;
import neureka.backend.standard.operations.linear.fast.array.operation.OperationUnary;
import neureka.backend.standard.operations.linear.fast.function.BinaryFunction;
import neureka.backend.standard.operations.linear.fast.function.FunctionSet;
import neureka.backend.standard.operations.linear.fast.function.PrimitiveFun;
import neureka.backend.standard.operations.linear.fast.function.UnaryFunction;
import neureka.backend.standard.operations.linear.fast.structure.Access1D;
import neureka.backend.standard.operations.linear.fast.type.NumberDefinition;

import java.util.Arrays;


/**
 * A one- and/or arbitrary-dimensional array of double.
 *
 */
public class F64Array extends PrimitiveArray {

    public static final Factory<Double> FACTORY = new Factory<Double>() {

        @Override
        public FunctionSet<Double> forFunctions() {
            return PrimitiveFun.getSet();
        }

    };

    public final double[] data;

    /**
     * Array not copied! No checking!
     */
    protected F64Array(final double[] data) {

        super(FACTORY, data.length);

        this.data = data;
    }

    protected F64Array(final int size) {

        super(FACTORY, size);

        data = new double[size];
    }

    @Override
    public boolean equals(final Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof F64Array)) {
            return false;
        }
        F64Array other = (F64Array) obj;
        return Arrays.equals(data, other.data);
    }

    @Override
    public void fillMatchingAccess(final Access1D<?> values) {
        if (values instanceof F64Array) {
            FillMatchingSingle.fill(data, ((F64Array) values).data);
        } else {
            FillMatchingSingle.fill(data, values);
        }
    }

    @Override
    public int size() {
        return data.length;
    }

    @Override
    protected void add(final int index, final Comparable<?> addend) {
        data[index] += NumberDefinition.doubleValue(addend);
    }

    @Override
    protected void add(final int index, final double addend) {
        data[index] += addend;
    }

    @Override
    protected void add(final int index, final float addend) {
        data[index] += addend;
    }

    protected final double[] copyOfData() {
        return COPY.copyOf(data);
    }

    @Override
    protected final double doubleValue(final int index) {
        return data[index];
    }

    @Override
    protected float floatValue(final int index) {
        return (float) data[index];
    }

    @Override
    protected final Double get(final int index) {
        return data[index];
    }

    @Override
    protected final void modify(final int first, final int limit, final int step, final Access1D<Double> left, final BinaryFunction<Double> function) {
        OperationBinary.invoke(data, first, limit, step, left, function, this);
    }

    @Override
    protected final void modify(final int first, final int limit, final int step, final BinaryFunction<Double> function, final Access1D<Double> right) {
        OperationBinary.invoke(data, first, limit, step, this, function, right);
    }

    @Override
    protected final void modify(final int first, final int limit, final int step, final UnaryFunction<Double> function) {
        OperationUnary.invoke(data, first, limit, step, this, function);
    }

    @Override
    protected final void set(final int index, final Comparable<?> value) {
        data[index] = NumberDefinition.doubleValue(value);
    }

    @Override
    protected final void set(final int index, final double value) {
        data[index] = value;
    }

    @Override
    protected final void set(final int index, final float value) {
        data[index] = value;
    }

}
