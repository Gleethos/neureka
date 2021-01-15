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

/**
 *  This interface represents instances of utility classes which represent numeric data types
 *  alongside useful methods which offer extensive conversion methods for various types
 *  of primitive or non-primitive data types and arrays of said types.
 *  Instances of concrete sub-types do not embody data types themselves,
 *  however they simply provide standardized methods which handle the
 *  type which the class represents.
 *
 * @param <TargetType>
 * @param <TargetArrayType>
 * @param <HolderType>
 * @param <HolderArrayType>
 */
public interface NumericType<TargetType, TargetArrayType, HolderType, HolderArrayType>
{
    /**
     *  This boolean value tells if the data-type represented
     *  by concrete instances of implementations of this interface
     *  is signed!
     *
     * @return The truth value which defines if the represented data-type is signed.
     */
    boolean signed();

    /**
     *
     * @return The number of bytes which it takes to represent the data-type.
     */
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


}
