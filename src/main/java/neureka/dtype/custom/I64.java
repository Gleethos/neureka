package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;

public class I64 extends AbstractNumericType<Long, long[]>
{

    public I64() { super(); }

    @Override
    public boolean signed() {
        return true;
    }

    @Override
    public int numberOfBytes() {
        return 8;
    }

    @Override
    public Class<Long> targetType() {
        return Long.class;
    }

    @Override
    public Class<long[]> targetArrayType() {
        return long[].class;
    }

    @Override
    public Long convert(byte[] bytes) {
        return null;
    }

    @Override
    public byte[] convert(Long number) {
        return new byte[0];
    }

    @Override
    public long[] readDataFrom(DataInput stream, int size) throws IOException {
        return new long[0];
    }

    @Override
    public void writeDataTo(DataOutput stream, long[] data) throws IOException {

    }

}
