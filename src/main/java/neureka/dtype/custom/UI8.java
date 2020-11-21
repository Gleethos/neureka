package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;
import neureka.utility.DataConverter;

import java.io.IOException;
import java.io.DataInput;

public class UI8 extends AbstractNumericType<Short, short[], Byte, byte[]>
{

    public UI8() {
        super();
        _set( double[].class, DataConverter.Utility::shortToDouble);
        _set( int[].class,    DataConverter.Utility::shortToInt);
        _set( float[].class,  DataConverter.Utility::shortToFloat);
        _set( long[].class,   DataConverter.Utility::shortToLong );
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
    public Class<Byte> foreignType() {
        return Byte.class;
    }

    @Override
    public Class<byte[]> foreignArrayType() {
        return byte[].class;
    }

    @Override
    public Short foreignBytesToTarget(byte[] bytes) {
        return //toTarget( bytes[ 0 ] );
        (short) Utility.unsignedByteArrayToInt(bytes);
    }

    @Override
    public Short toTarget(Byte original) {
        return (short) Byte.toUnsignedInt( original );
        //return (short) (original & 0xFF);
    }

    @Override
    public byte[] targetToForeignBytes(Short number) {
        return new byte[]{(byte)(number & 0xFF)};
    }

    @Override
    public short[] readAndConvertDataFrom(DataInput stream, int size) throws IOException {
        short[] data = new short[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[ i ] = foreignBytesToTarget(_data);
        }
        return data;
    }

    @Override
    public byte[] readForeignDataFrom(DataInput stream, int size) throws IOException {
        byte[] data = new byte[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[ i ] = _data[0];
        }
        return data;
    }

}
