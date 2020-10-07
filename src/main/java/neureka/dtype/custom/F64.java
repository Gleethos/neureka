package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.nio.ByteBuffer;
import java.util.Iterator;

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
        return ByteBuffer.wrap(bytes).getDouble();
    }

    @Override
    public byte[] convert(Double number) {
        long data = Double.doubleToRawLongBits(number);
        return new byte[] {
                (byte) ((data >> 56) & 0xff),
                (byte) ((data >> 48) & 0xff),
                (byte) ((data >> 40) & 0xff),
                (byte) ((data >> 32) & 0xff),
                (byte) ((data >> 24) & 0xff),
                (byte) ((data >> 16) & 0xff),
                (byte) ((data >> 8) & 0xff),
                (byte) ((data >> 0) & 0xff),
        };
    }

    @Override
    public double[] readDataFrom(DataInput stream, int size) throws IOException {
        double[] data = new double[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[i] = convert(_data);
        }
        return data;
    }

}
