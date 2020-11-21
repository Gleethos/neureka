package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.nio.ByteBuffer;

public class I32 extends AbstractNumericType<Integer, int[], Integer, int[]>
{

    public I32() {
        super();
    }

    @Override
    public boolean signed() {
        return true;
    }

    @Override
    public int numberOfBytes() {
        return 4;
    }

    @Override
    public Class<Integer> targetType() {
        return Integer.class;
    }

    @Override
    public Class<int[]> targetArrayType() {
        return int[].class;
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
    public Integer foreignBytesToTarget(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getInt();
        //return Utility.unsignedByteArrayToInt(_data);
    }

    @Override
    public Integer toTarget(Integer original) {
        return original;
    }

    @Override
    public byte[] targetToForeignBytes(Integer number) {
        return new byte[] {
                (byte)((number >> 24) & 0xff),
                (byte)((number >> 16) & 0xff),
                (byte)((number >> 8) & 0xff),
                (byte)((number >> 0) & 0xff),
        };
    }

    @Override
    public int[] readAndConvertDataFrom( DataInput stream, int size ) throws IOException {
        return _readData( stream, size );
    }

    @Override
    public int[] readForeignDataFrom(DataInput stream, int size ) throws IOException {
        return _readData( stream, size );
    }

    private int[] _readData( DataInput stream, int size ) throws IOException {
        int[] data = new int[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[ i ] = foreignBytesToTarget(_data);
        }
        return data;
    }

}
