package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;

public class F64 extends AbstractNumericType<Double, double[]>
{

    public F64() { super(); }

    @Override
    public boolean signed() {
        return true;
    }

    @Override
    public int numberOfBytes() {
        return 8;
    }

    @Override
    public Class<Double> targetType() {
        return Double.class;
    }

    @Override
    public Class<double[]> targetArrayType() {
        return double[].class;
    }

    @Override
    public Double convert(byte[] bytes) {
        return null;
    }

    @Override
    public byte[] convert(Double number) {
        return new byte[0];
    }

    @Override
    public double[] readDataFrom(DataInput stream, int size) throws IOException {
        return new double[0];
    }

    @Override
    public void writeDataTo(DataOutput stream, double[] data) throws IOException {

    }

}
