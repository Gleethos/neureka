package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;
import neureka.common.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

public final class I32 extends AbstractNumericType<Integer, int[], Integer, int[]>
{

    public I32() { super(); }

    @Override public boolean signed() { return true; }

    @Override public int numberOfBytes() { return 4; }

    @Override
    public Class<Integer> targetType() { return Integer.class; }

    @Override
    public Class<int[]> targetArrayType() { return int[].class; }

    @Override public Class<Integer> holderType() { return Integer.class; }

    @Override public Class<int[]> holderArrayType() { return int[].class; }

    @Override
    public Integer foreignHolderBytesToTarget( byte[] bytes ) {
        return ByteBuffer.wrap(bytes).getInt();
        //return Utility.unsignedByteArrayToInt(_data);
    }

    @Override
    public Integer toTarget( Integer original ) {
        return original;
    }

    @Override
    public byte[] targetToForeignHolderBytes( Integer number ) {
        return new byte[] {
                (byte)((number >> 24) & 0xff),
                (byte)((number >> 16) & 0xff),
                (byte)((number >> 8) & 0xff),
                (byte)((number >> 0) & 0xff),
        };
    }

    @Override
    public int[] readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException {
        return _readData( stream, size );
    }

    @Override
    public <T> int[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        return new int[0];
    }

    @Override
    public int[] readForeignDataFrom(DataInput stream, int size ) throws IOException {
        return _readData( stream, size );
    }

    @Override
    public <T> int[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        return new int[0];
    }

    @Override
    public Integer convertToHolder( Object from ) {
        if ( Byte.class.equals( from.getClass() ) )
            return ( (Byte) from ).intValue();
        else if ( Integer.class.equals( from.getClass() ) )
            return ( (Integer) from ).intValue();
        else if ( Double.class.equals( from.getClass() ) )
            return ( (Double) from ).intValue();
        else if ( Short.class.equals( from.getClass() ) )
            return ( (Short) from ).intValue();
        else if ( Long.class.equals( from.getClass() ) )
            return ( (Long) from ).intValue();
        else if ( Float.class.equals( from.getClass() ) )
            return ( (Float) from ).intValue();
        else
            return null;
    }

    @Override
    public int[] convertToHolderArray( Object from ) {
        return new int[0];
    }

    @Override
    public Integer convertToTarget( Object from ) {
        return convertToHolder( from );
    }

    @Override
    public int[] convertToTargetArray( Object from ) {
        return DataConverter.get().convert( from, int[].class );
    }

    private int[] _readData( DataInput stream, int size ) throws IOException {
        int[] data = new int[size];
        byte[] bytes = new byte[ this.numberOfBytes() ];
        for ( int i=0; i<size; i++ ) {
            stream.readFully( bytes );
            data[ i ] = foreignHolderBytesToTarget( bytes );
        }
        return data;
    }

}
