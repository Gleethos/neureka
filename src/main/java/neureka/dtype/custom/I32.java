package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.nio.ByteBuffer;

public class I32 extends AbstractNumericType<Integer, int[]>
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
    public Integer convert(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getInt();
        //return Utility.unsignedByteArrayToInt(_data);
    }

    @Override
    public byte[] convert(Integer number) {
        return new byte[] {
                (byte)((number >> 24) & 0xff),
                (byte)((number >> 16) & 0xff),
                (byte)((number >> 8) & 0xff),
                (byte)((number >> 0) & 0xff),
        };
    }

    @Override
    public int[] readDataFrom(DataInput stream, int size) throws IOException {
        int[] data = new int[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[ i ] = convert(_data);
        }
        return data;
    }


}
