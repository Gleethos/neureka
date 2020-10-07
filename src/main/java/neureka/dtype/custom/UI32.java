package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteBuffer;

public class UI32 extends AbstractNumericType<Long, long[]>
{

    @Override
    public boolean signed() {
        return false;
    }

    @Override
    public int numberOfBytes() {
        return 4;
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
        int asInt = ByteBuffer.wrap(bytes).getInt();
        return ( asInt >= 0 )
                    ? (long) asInt
                    : (long)(0x40000000 + asInt);
    }

    @Override
    public byte[] convert(Long number) {
        final ByteBuffer buf = ByteBuffer.allocate(8);
        buf.putLong( number );
        byte[] b = buf.array();
        return new byte[]{ b[4], b[5], b[6], b[7] };
    }

    @Override
    public long[] readDataFrom(DataInput stream, int size) throws IOException {
        return new long[0];
    }


    @Override
    public <T> T convert(long[] from, Class<T> to) {
        return null;
    }
}
