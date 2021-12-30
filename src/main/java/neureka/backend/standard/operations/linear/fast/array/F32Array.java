/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.array;

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
public class F32Array extends PrimitiveArray {

    public static final Factory<Double> FACTORY = new Factory<Double>() {

        @Override
        public FunctionSet<Double> forFunctions() {
            return PrimitiveFun.getSet();
        }

    };

    public final float[] data;

    /**
     * Array not copied! No checking!
     */
    protected F32Array(final float[] data) {

        super(FACTORY, data.length);

        this.data = data;
    }

    protected F32Array(final int size) {

        super(FACTORY, size);

        data = new float[size];
    }

    @Override
    public boolean equals(final Object obj) {
        if (this == obj) {
            return true;
        }
        if (!super.equals(obj) || !(obj instanceof F32Array)) {
            return false;
        }
        F32Array other = (F32Array) obj;
        if (!Arrays.equals(data, other.data)) {
            return false;
        }
        return true;
    }

    @Override
    public void fillMatchingAccess(final Access1D<?> values) {
        if (values instanceof F32Array) {
            FillMatchingSingle.fill(data, ((F32Array) values).data);
        } else {
            FillMatchingSingle.fill(data, values);
        }
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = prime * result + Arrays.hashCode(data);
        return result;
    }

    @Override
    public int size() {
        return data.length;
    }

    @Override
    protected void add(final int index, final Comparable<?> addend) {
        data[index] += NumberDefinition.floatValue(addend);
    }

    @Override
    protected void add(final int index, final double addend) {
        data[index] += (float) addend;
    }

    @Override
    protected void add(final int index, final float addend) {
        data[index] += addend;
    }

    @Override
    protected final double doubleValue(final int index) {
        return data[index];
    }

    @Override
    protected float floatValue(final int index) {
        return data[index];
    }

    @Override
    protected final Double get(final int index) {
        return Double.valueOf(data[index]);
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
        data[index] = NumberDefinition.floatValue(value);
    }

    @Override
    protected final void set(final int index, final double value) {
        data[index] = (float) value;
    }

    @Override
    protected final void set(final int index, final float value) {
        data[index] = value;
    }

}
