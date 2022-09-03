package neureka.dtype.custom;

import neureka.common.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

public final class I16 extends AbstractNumericType<Short, short[], Short, short[]>
{
    private final ByteBuffer buffer = ByteBuffer.allocate(Short.BYTES);

    public I16() { super(); }

    @Override
    public boolean signed() { return true; }

    @Override
    public int numberOfBytes() { return 2; }

    @Override
    public Class<Short> targetType() { return Short.class; }

    @Override
    public Class<short[]> targetArrayType() { return short[].class; }

    @Override public Class<Short> holderType() { return Short.class; }

    @Override public Class<short[]> holderArrayType() { return short[].class; }

    @Override
    public Short foreignHolderBytesToTarget( byte[] bytes ) {
        buffer.put(bytes, 0, bytes.length);
        buffer.flip();//need flip
        return buffer.getShort();
    }

    @Override
    public Short toTarget(Short original) {
        return original;
    }

    @Override
    public byte[] targetToForeignHolderBytes(Short number) {
        buffer.putShort(0, number);
        return buffer.array();
    }

    @Override
    public short[] readAndConvertForeignDataFrom(DataInput stream, int size ) throws IOException {
        return _readData( stream, size );
    }

    @Override
    public <T> short[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public short[] readForeignDataFrom(DataInput stream, int size ) throws IOException {
        return _readData( stream, size );
    }

    @Override
    public <T> short[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public Short convertToHolder( Object from ) {
        if ( Byte.class.equals( from.getClass() ) )
            return ( (Byte) from ).shortValue();
        else if ( Integer.class.equals( from.getClass() ) )
            return ( (Integer) from ).shortValue();
        else if ( Double.class.equals( from.getClass() ) )
            return ( (Double) from ).shortValue();
        else if ( Short.class.equals( from.getClass() ) )
            return ( (Short) from );
        else if ( Long.class.equals( from.getClass() ) )
            return ( (Long) from ).shortValue();
        else if ( Float.class.equals( from.getClass() ) )
            return ( (Float) from ).shortValue();
        else
            return null;
    }

    @Override
    public short[] convertToHolderArray( Object from ) {
        throw new UnsupportedOperationException("Not implemented yet!");
    }

    @Override
    public Short convertToTarget( Object from ) {
        return convertToHolder( from );
    }

    @Override
    public short[] convertToTargetArray( Object from ) {
        return DataConverter.get().convert( from, short[].class );
    }

    private short[] _readData( DataInput stream, int size ) throws IOException {
        short[] data = new short[size];
        byte[] bytes = new byte[ this.numberOfBytes() ];
        for ( int i=0; i<size; i++ ) {
            stream.readFully( bytes );
            data[ i ] = foreignHolderBytesToTarget( bytes );
        }
        return data;
    }

}
