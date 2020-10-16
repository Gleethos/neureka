package neureka.dtype;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.Iterator;

public interface NumericType<TargetType, ArrayType>
{
    boolean signed();

    int numberOfBytes();

    Class<TargetType> targetType();

    Class<ArrayType> targetArrayType();

    TargetType convert(byte[] bytes);

    byte[] convert(TargetType number);

    ArrayType readDataFrom(DataInput stream, int size) throws IOException;

    void writeDataTo(DataOutput stream, Iterator<TargetType> iterator) throws IOException;

    <T> T convert(ArrayType from, Class<T> to);

    class Utility
    {
        public static int unsignedByteArrayToInt(byte[] b)
        { // This views the given bytes as unsigned!
            if ( b.length == 4 ) return b[ 0 ] << 24 | (b[1] & 0xff) << 16 | (b[2] & 0xff) << 8 | (b[3] & 0xff);
            else if ( b.length == 2 ) return 0x00 << 24 | 0x00 << 16 | (b[ 0 ] & 0xff) << 8 | (b[1] & 0xff);
            else if ( b.length == 1 ) return ((int)b[ 0 ]) & 0xFF;
            return 0;
        }

        public static byte[] integerToByteArray(int i) {
            return ByteBuffer.allocate(4).putInt( i ).array();
        }

    }

}
