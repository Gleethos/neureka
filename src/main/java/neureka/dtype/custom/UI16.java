package neureka.dtype.custom;

import neureka.dtype.AbstractNumericType;
import neureka.utility.DataConverter;

import java.io.IOException;
import java.io.DataInput;
import java.nio.ByteBuffer;
import java.util.Iterator;

public final class UI16 extends AbstractNumericType<Integer, int[], Short, short[]>
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
    public Class<Short> holderType() {
        return null;
    }

    @Override
    public Class<short[]> holderArrayType() {
        return null;
    }

    @Override
    public Integer foreignHolderBytesToTarget(byte[] b ) {
        return
                0x00 << 24 |
                0x00 << 16 |
                (b[ 0 ] & 0xff) << 8 |
                (b[ 1 ] & 0xff);
    }

    @Override
    public Integer toTarget( Short original ) {
        return Short.toUnsignedInt( original );
    }

    @Override
    public byte[] targetToForeignHolderBytes(Integer number ) {
        final ByteBuffer buf = ByteBuffer.allocate(2);
        buf.putShort( (short) number.intValue() );
        return buf.array();
    }

    @Override
    public int[] readAndConvertForeignDataFrom(DataInput stream, int size )  throws IOException {
        int[] data = new int[ size ];
        byte[] bytes = new byte[ this.numberOfBytes() ];
        for ( int i=0; i<size; i++ ) {
            stream.readFully( bytes );
            data[ i ] = foreignHolderBytesToTarget( bytes );
        }
        return data;
    }

    @Override
    public <T> int[] readAndConvertForeignDataFrom( Iterator<T> iterator, int size ) {
        return new int[0];
    }

    @Override
    public short[] readForeignDataFrom( DataInput stream, int size ) throws IOException {
        return new short[0];
    }

    @Override
    public <T> short[] readForeignDataFrom( Iterator<T> iterator, int size ) {
        return new short[0];
    }

    @Override
    public Short convertToHolder( Object from ) {
        return null;
    }

    @Override
    public short[] convertToHolderArray( Object from ) {
        return new short[0];
    }

    @Override
    public Integer convertToTarget( Object from ) {
        return null;
    }

    @Override
    public int[] convertToTargetArray( Object from ) {
        return DataConverter.instance().convert( from, int[].class );
    }


}
