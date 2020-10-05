package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;

public class F32 extends AbstractNumericType<Float, float[]>
{
    public F32() { super(); }

    @Override
    public boolean signed() {
        return true;
    }

    @Override
    public int numberOfBytes() {
        return 4;
    }

    @Override
    public Class<Float> targetType() {
        return Float.class;
    }

    @Override
    public Class<float[]> targetArrayType() {
        return float[].class;
    }

    @Override
    public Float convert(byte[] bytes) {
        return null;
    }

    @Override
    public byte[] convert(Float number) {
        return new byte[0];
    }

    @Override
    public float[] readDataFrom(DataInput stream, int size) throws IOException {
        return new float[0];
    }

    @Override
    public void writeDataTo(DataOutput stream, float[] data) throws IOException {

    }

}
