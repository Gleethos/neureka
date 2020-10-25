package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.math.BigInteger;
import java.nio.ByteBuffer;

public class UI16 extends AbstractNumericType<Integer, int[]>
{

    public UI16() {
        super();
    }

    @Override
    public boolean signed() {
        return false;
    }

    @Override
    public int numberOfBytes() {
        return 2;
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
    public Integer convert(byte[] b) {
        return 0x00 << 24 | 0x00 << 16 | (b[ 0 ] & 0xff) << 8 | (b[ 1 ] & 0xff);
        //return Utility.unsignedByteArrayToInt(bytes);
    }

    @Override
    public byte[] convert(Integer number) {
        final ByteBuffer buf = ByteBuffer.allocate(2);
        buf.putShort( (short) number.intValue() );
        return buf.array();
    }

    @Override
    public int[] readDataFrom(DataInput stream, int size)  throws IOException {
        int[] data = new int[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[ i ] = convert(_data);
        }
        return data;
    }


}
