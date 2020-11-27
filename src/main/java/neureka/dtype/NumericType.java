/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   _   _                           _   _______
  | \ | |                         (_) |__   __|
  |  \| |_   _ _ __ ___   ___ _ __ _  ___| |_   _ _ __   ___
  | . ` | | | | '_ ` _ \ / _ \ '__| |/ __| | | | | '_ \ / _ \
  | |\  | |_| | | | | | |  __/ |  | | (__| | |_| | |_) |  __/
  |_| \_|\__,_|_| |_| |_|\___|_|  |_|\___|_|\__, | .__/ \___|
                                             __/ | |
                                            |___/|_|

*/

package neureka.dtype;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.Iterator;

public interface NumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
{
    boolean signed();

    int numberOfBytes();

    Class<TargetType> targetType();

    Class<TargetArrayType> targetArrayType();

    Class<HolderType> holderType();

    Class<HolderArrayType> holderArrayType();

    Class<NumericType<TargetType, TargetArrayType, TargetType, TargetArrayType>> getNumericTypeTarget();

    TargetType foreignHolderBytesToTarget( byte[] bytes );

    TargetType toTarget( HolderType original );

    byte[] targetToForeignHolderBytes(TargetType number );

    TargetArrayType readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException;

    <T> TargetArrayType readAndConvertForeignDataFrom( Iterator<T> iterator, int size );

    HolderArrayType readForeignDataFrom (DataInput stream, int size ) throws IOException;

    <T> HolderArrayType readForeignDataFrom( Iterator<T> iterator, int size );

    void writeDataTo( DataOutput stream, Iterator<TargetType> iterator ) throws IOException;

    HolderType convertToHolder( Object from );

    HolderArrayType convertToHolderArray( Object from );

    TargetType convertToTarget( Object from );

    TargetArrayType convertToTargetArray( Object from );

    class Utility
    {
        public static int unsignedByteArrayToInt( byte[] b )
        { // This views the given bytes as unsigned!
            if ( b.length == 4 ) return b[ 0 ] << 24 | (b[ 1 ] & 0xff) << 16 | (b[ 2 ] & 0xff) << 8 | (b[3] & 0xff);
            else if ( b.length == 2 ) return 0x00 << 24 | 0x00 << 16 | (b[ 0 ] & 0xff) << 8 | (b[ 1 ] & 0xff);
            else if ( b.length == 1 ) return ((int)b[ 0 ]) & 0xFF;
            return 0;
        }

        public static byte[] integerToByteArray( int i ) {
            return ByteBuffer.allocate(4).putInt( i ).array();
        }

    }

}
