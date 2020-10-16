package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class I16 extends AbstractNumericType<Short, short[]>
{
    private final ByteBuffer buffer = ByteBuffer.allocate(Short.BYTES);

    public I16(){ super(); }

    @Override
    public boolean signed() {
        return true;
    }

    @Override
    public int numberOfBytes() {
        return 2;
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
    public Short convert(byte[] bytes) {
        buffer.put(bytes, 0, bytes.length);
        buffer.flip();//need flip
        return buffer.getShort();
        //return ByteBuffer.wrap(bytes).order(ByteOrder.).getShort();
    }

    @Override
    public byte[] convert(Short number) {
        buffer.putShort(0, number);
        return buffer.array();
        //return new byte[]{(byte)(number & 0xff),(byte)((number >> 8) & 0xff)};
    }

    @Override
    public short[] readDataFrom(DataInput stream, int size) throws IOException {
        short[] data = new short[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[ i ] = convert(_data);
        }
        return data;
    }


}
