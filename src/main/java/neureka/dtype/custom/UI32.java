package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteBuffer;

public class UI32 extends AbstractNumericType<Long, long[], Integer, int[]>
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
    public Class<Integer> foreignType() {
        return Integer.class;
    }

    @Override
    public Class<int[]> foreignArrayType() {
        return int[].class;
    }

    @Override
    public Long foreignBytesToTarget(byte[] bytes) {
        //int asInt = ByteBuffer.wrap(bytes).getInt();
        //System.out.println("... "+asInt);
        //return ( asInt >= 0 )
        //            ? (long) asInt
        //            : (((long)0x40000000) + (long)asInt);
        return
            //0x00L << 64 |
            //0x00L << 56 |
            //0x00L << 48 |
            //0x00L << 40 |
            //0x00L << 32 |
            ((long)( bytes[ 0 ] & 0xff )) << 24 |
            ((long)( bytes[ 1 ] & 0xff )) << 16 |
            ( bytes[ 2 ] & 0xff ) << 8 |
            ( bytes[ 3 ] & 0xff );
    }

    @Override
    public Long toTarget(Integer original) {
        return Integer.toUnsignedLong( original );
    }

    @Override
    public byte[] targetToForeignBytes(Long number) {
        final ByteBuffer buf = ByteBuffer.allocate(8);
        buf.putLong( number );
        byte[] b = buf.array();
        return new byte[]{ b[4], b[5], b[6], b[7] };
    }

    @Override
    public long[] readAndConvertDataFrom(DataInput stream, int size) throws IOException {
        return new long[ 0 ];
    }

    @Override
    public int[] readForeignDataFrom(DataInput stream, int size) throws IOException {
        return new int[0];
    }


    @Override
    public <T> T convert(long[] from, Class<T> to) {
        return null;
    }
}
