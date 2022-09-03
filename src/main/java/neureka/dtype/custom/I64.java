package neureka.dtype.custom;

import neureka.common.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

public final class I64 extends AbstractNumericType<Long, long[], Long, long[]>
{
    private final ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);

    public I64() { super(); }

    @Override public boolean signed() { return true; }

    @Override public int numberOfBytes() { return 8; }

    @Override public Class<Long> targetType() { return Long.class; }

    @Override public Class<long[]> targetArrayType() { return long[].class; }

    @Override public Class<Long> holderType() { return Long.class; }

    @Override
    public Class<long[]> holderArrayType() { return long[].class; }

    @Override
    public Long foreignHolderBytesToTarget( byte[] bytes ) {
        buffer.put(bytes, 0, bytes.length);
        buffer.flip();//need flip
        return buffer.getLong();
        //return ByteBuffer.wrap(bytes).getLong();
    }

    @Override
    public Long toTarget(Long original) {
        return original;
    }

    @Override
    public byte[] targetToForeignHolderBytes(Long number) {
        buffer.putLong(0, number);
        return buffer.array();
    }

    @Override
    public long[] readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public <T> long[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        long[] data = new long[size];
        for ( int i = 0; i < size; i++ ) data[ i ] = convertToHolder( iterator.next() );
        return data;
    }

    @Override
    public long[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public <T> long[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        return readAndConvertForeignDataFrom( iterator, size );
    }

    @Override
    public Long convertToHolder( Object from ) {
        return DataConverter.get().convert( from, Long.class );
    }

    @Override
    public long[] convertToHolderArray( Object from ) {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public Long convertToTarget( Object from ) {
        return convertToHolder( from );
    }

    @Override
    public long[] convertToTargetArray( Object from ) {
        return DataConverter.get().convert( from, long[].class );
    }

}
