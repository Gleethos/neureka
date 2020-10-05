package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;

import java.io.IOException;
import java.io.DataInput;
import java.io.DataOutput;
import java.nio.ByteBuffer;

public class I16 extends AbstractNumericType<Short, short[]>
{

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
        return ByteBuffer.wrap(bytes).getShort();
    }

    @Override
    public byte[] convert(Short number) {
        return new byte[]{(byte)(number & 0xff),(byte)((number >> 8) & 0xff)};
    }

    @Override
    public short[] readDataFrom(DataInput stream, int size) throws IOException {
        short[] data = new short[size];
        for ( int i=0; i<size; i++ ) {
            stream.readFully(_data);
            data[i] = convert(_data);
        }
        return data;
    }

    @Override
    public void writeDataTo(DataOutput stream, short[] data) throws IOException {

    }


}
