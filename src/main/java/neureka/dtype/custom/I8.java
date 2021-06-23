package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;
import neureka.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.util.Iterator;

/**
 *  The following abstract class implements some basic logic which
 *  is applicable across all final concrete classes extending this abstract one.
 *
 */
public final class I8 extends AbstractNumericType<Byte, byte[], Byte, byte[]>
{
    public I8() {
        super();
    }

    @Override
    public boolean signed() {
        return true;
    }

    @Override
    public int numberOfBytes() {
        return 1;
    }

    @Override
    public Class<Byte> targetType() {
        return Byte.class;
    }

    @Override
    public Class<byte[]> targetArrayType() {
        return byte[].class;
    }

    @Override
    public Class<Byte> holderType() {
        return Byte.class;
    }

    @Override
    public Class<byte[]> holderArrayType() {
        return byte[].class;
    }

    @Override
    public Byte foreignHolderBytesToTarget( byte[] bytes ) {
        return bytes[ 0 ];
    }

    @Override
    public Byte toTarget( Byte original ) {
        return original;
    }

    @Override
    public byte[] targetToForeignHolderBytes( Byte number ) {
        return new byte[]{number};
    }

    @Override
    public byte[] readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException {
        byte[] bytes = new byte[size];
        stream.readFully(bytes, size, size);
        return bytes;
    }

    @Override
    public <T> byte[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        return new byte[0];
    }

    @Override
    public byte[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        byte[] bytes = new byte[size];
        stream.readFully(bytes, size, size);
        return bytes;
    }

    @Override
    public <T> byte[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        return new byte[0];
    }

    @Override
    public Byte convertToHolder( Object from ) {
        if ( Byte.class.equals( from.getClass() ) )
            return ( (Byte) from );
        else if ( Integer.class.equals( from.getClass() ) )
            return ( (Integer) from ).byteValue();
        else if ( Double.class.equals( from.getClass() ) )
            return ( (Double) from ).byteValue();
        else if ( Short.class.equals( from.getClass() ) )
            return ( (Short) from ).byteValue();
        else if ( Long.class.equals( from.getClass() ) )
            return ( (Long) from ).byteValue();
        else if ( Float.class.equals( from.getClass() ) )
            return ( (Float) from ).byteValue();
        else
            return null;
    }

    @Override
    public byte[] convertToHolderArray( Object from ) {
        return new byte[0];
    }

    @Override
    public Byte convertToTarget( Object from ) {
        return convertToHolder( from );
    }

    @Override
    public byte[] convertToTargetArray( Object from ) {
        return DataConverter.instance().convert( from, byte[].class );
    }


}
