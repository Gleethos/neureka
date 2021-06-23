package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;
import neureka.utility.DataConverter;

import java.io.DataInput;
import java.io.IOException;
import java.util.Iterator;

public final class UI8 extends AbstractNumericType<Short, short[], Byte, byte[]>
{

    public UI8() {
        super();
    }

    @Override
    public boolean signed() {
        return false;
    }

    @Override
    public int numberOfBytes() {
        return 1;
    }

    @Override
    public Class<Short> targetType() {
        return Short.class;
    }

    @Override
    public Class<short[]> targetArrayType() {
        return short[].class;
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
    public Short foreignHolderBytesToTarget( byte[] bytes ) {
        return (short) (((int)bytes[ 0 ]) & 0xFF);
    }


    @Override
    public Short toTarget(Byte original) {
        return (short) Byte.toUnsignedInt( original );
        //return (short) (original & 0xFF);
    }

    @Override
    public byte[] targetToForeignHolderBytes(Short number) {
        return new byte[]{(byte)(number & 0xFF)};
    }

    @Override
    public short[] readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException {
        short[] data = new short[size];
        byte[] bytes = new byte[ this.numberOfBytes() ];
        for ( int i=0; i<size; i++ ) {
            stream.readFully( bytes );
            data[ i ] = foreignHolderBytesToTarget( bytes );
        }
        return data;
    }

    @Override
    public <T> short[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        return new short[0];
    }

    @Override
    public byte[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        byte[] data = new byte[ size ];
        byte[] bytes = new byte[ this.numberOfBytes() ];
        for ( int i=0; i<size; i++ ) {
            stream.readFully( bytes );
            data[ i ] = bytes[ 0 ];
        }
        return data;
    }

    @Override
    public <T> byte[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        return new byte[0];
    }

    @Override
    public Byte convertToHolder( Object from ) {
        return null;
    }

    @Override
    public byte[] convertToHolderArray( Object from ) {
        return new byte[0];
    }

    @Override
    public Short convertToTarget( Object from ) {
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
    public short[] convertToTargetArray( Object from ) {
        return DataConverter.instance().convert( from, short[].class );
    }

}
