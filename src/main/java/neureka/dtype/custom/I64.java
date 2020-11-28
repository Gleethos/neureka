package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;
import neureka.utility.DataConverter;

import java.io.IOException;
import java.io.DataInput;
import java.nio.ByteBuffer;
import java.util.Iterator;

public class I64 extends AbstractNumericType<Long, long[], Long, long[]>
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
    public Class<Long> holderType() {
        return Long.class;
    }

    @Override
    public Class<long[]> holderArrayType() {
        return long[].class;
    }

    @Override
    public Long foreignHolderBytesToTarget(byte[] bytes) {
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
        return new long[ 0 ];
    }

    @Override
    public <T> long[] readAndConvertForeignDataFrom(Iterator<T> iterator, int size) {
        return new long[0];
    }

    @Override
    public long[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        return new long[0];
    }

    @Override
    public <T> long[] readForeignDataFrom(Iterator<T> iterator, int size) {
        return new long[0];
    }

    @Override
    public Long convertToHolder(Object from) {
        if ( Byte.class.equals( from.getClass() ) )
            return ( (Byte) from ).longValue();
        else if ( Integer.class.equals( from.getClass() ) )
            return ( (Integer) from ).longValue();
        else if ( Double.class.equals( from.getClass() ) )
            return ( (Double) from ).longValue();
        else if ( Short.class.equals( from.getClass() ) )
            return ( (Short) from ).longValue();
        else if ( Long.class.equals( from.getClass() ) )
            return ( (Long) from );
        else if ( Float.class.equals( from.getClass() ) )
            return ( (Float) from ).longValue();
        else
            return null;
    }

    @Override
    public long[] convertToHolderArray(Object from) {
        return new long[0];
    }

    @Override
    public Long convertToTarget(Object from) {
        return convertToHolder(from);
    }

    @Override
    public long[] convertToTargetArray(Object from) {
        return DataConverter.instance().convert( from, long[].class );
    }

}
