package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;
import neureka.common.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

public final class UI32 extends AbstractNumericType<Long, long[], Integer, int[]>
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
    public Class<Integer> holderType() {
        return Integer.class;
    }

    @Override
    public Class<int[]> holderArrayType() {
        return int[].class;
    }

    @Override
    public Long foreignHolderBytesToTarget( byte[] bytes ) {
        return
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
    public byte[] targetToForeignHolderBytes(Long number) {
        final ByteBuffer buf = ByteBuffer.allocate(8);
        buf.putLong( number );
        byte[] b = buf.array();
        return new byte[]{ b[4], b[5], b[6], b[7] };
    }

    @Override
    public long[] readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException {
        return new long[ 0 ];
    }

    @Override
    public <T> long[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        return new long[0];
    }

    @Override
    public int[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        return new int[0];
    }

    @Override
    public <T> int[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        return new int[0];
    }

    @Override
    public Integer convertToHolder( Object from ) {
        return null;
    }

    @Override
    public int[] convertToHolderArray( Object from ) {
        return new int[0];
    }

    @Override
    public Long convertToTarget( Object from ) {
        return null;
    }

    @Override
    public long[] convertToTargetArray( Object from ) {
        return DataConverter.get().convert( from, long[].class );
    }


}
