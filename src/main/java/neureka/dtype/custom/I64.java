package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.math.BigInteger;
import java.nio.ByteBuffer;

public class I64 extends AbstractNumericType<Long, long[]>
{
    private final ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);

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
        buffer.put(bytes, 0, bytes.length);
        buffer.flip();//need flip
        return buffer.getLong();
        //return ByteBuffer.wrap(bytes).getLong();
    }

    @Override
    public byte[] convert(Long number) {
        buffer.putLong(0, number);
        return buffer.array();
        //return new byte[] {
        //        (byte) (number >> 0),
        //        (byte) (number >> 8),
        //        (byte) (number >> 16),
        //        (byte) (number >> 24),
        //        (byte) (number >> 32),
        //        (byte) (number >> 40),
        //        (byte) (number >> 48),
        //        (byte) (number >> 56)
        //};
    }

    @Override
    public long[] readDataFrom(DataInput stream, int size) throws IOException {
        return new long[ 0 ];
    }

}
