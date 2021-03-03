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

    An interface defining common functionality for
    numeric type utility / representation...

*/

package neureka.dtype;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.Iterator;

/**
 *  This interface enables "Polymorphic" utility by defining common functionalities
 *  used for handling various numeric types.
 *  Implementations of this interface are utility classes which represent the numeric data types
 *  specified by the class level type parameters, and offer an extensive set of conversion methods for them.
 *
 *  Instances of concrete sub-types do not embody data types themselves,
 *  they simply provide standardized methods which handle the type represented by the class.
 *
 * @param <TargetType> The target type is the targeted JVM data-type which can represent the holder type.
 * @param <TargetArrayType> The target array type is the targeted JVM array data-type which can represent the holder array type.
 * @param <HolderType> The holder type is the JVM type which can hold the data but not necessarily represent it (int cant represent uint).
 * @param <HolderArrayType> The holder array type is the JVM array type which can hold the data but not necessarily represent it (int[] cant represent uint[]).
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
     * @return The number of bytes which it takes to represent the data-type.
     */
    int numberOfBytes();

    /**
     *  The target type is the targeted JVM data-type which can represent the holder type.
     *  This method returns the class object of this target type.
     *
     * @return The targeted JVM data-type class which can represent the holder type.
     */
    Class<TargetType> targetType();

    /**
     *  The target type is the targeted JVM data-type which can represent the holder type.
     *  This method returns the class object of an array of this target type.
     *
     * @return The targeted JVM data-type array class which can represent the holder array type.
     */
    Class<TargetArrayType> targetArrayType();

    /**
     *  The holder type is the JVM type which can hold the data but not necessarily represent it (int cant represent uint).
     *  This method returns the class object of this holder type.
     *
     * @return The holder type class which can hold the data type but not necessarily represent it (int cant represent uint).
     */
    Class<HolderType> holderType();

    /**
     *  The holder array type is the JVM type which can hold the data but not necessarily represent it (int[] cant represent uint[]).
     *  This method returns the array class object of an array of this holder type.
     *
     * @return The holder array type class which can hold the data type array but not necessarily represent it (int[] cant represent uint[]).
     */
    Class<HolderArrayType> holderArrayType();

    /**
     *  This method returns the NumericType representation of the target type of this class.
     *  An example would be the UI32 (32 bit integer) which can be represented by
     *  the NumericType implementation I64 (64 bit integer).
     *  If this NumericType representation can represent itself, meaning its target type is also the holder
     *  type like for instance I32, then this method will simply return its own class!
     *
     * @return The NumericType representation of the target type class which can represent the data type of this class.
     */
    Class<NumericType<TargetType, TargetArrayType, TargetType, TargetArrayType>> getNumericTypeTarget();

    /**
     * @param bytes The raw bytes of the holder type.
     * @return An instance of a target type built based on the provided holder bytes.
     */
    TargetType foreignHolderBytesToTarget( byte[] bytes );

    /**
     * @param original An instance of a holder type which ought to be converted to an instance of the target type.
     * @return An instance of the target type converted from the provided holder type.
     */
    TargetType toTarget( HolderType original );

    /**
     * @param number An instance of a target type which ought to be converted to holder bytes.
     * @return Holder bytes converted from an instance of the target type.
     */
    byte[] targetToForeignHolderBytes(TargetType number );

    /**
     *  This method expects the provided stream to spit out bytes which can be read as holder type elements.
     *  It then ought to convert these to an array of target types of the specified size.
     *
     * @param stream A DataInput stream whose data ought to be read as holder type elements.
     * @param size The number of elements which ought to be read.
     * @return An array of target types converted from the holder type elements read from the stream.
     * @throws IOException If reading from the stream was not successful.
     */
    TargetArrayType readAndConvertForeignDataFrom( DataInput stream, int size ) throws IOException;

    /**
     *  This method expects the provided iterator to return elements which can be read as holder type elements.
     *  It then ought to convert these to an array of target types of the specified size.
     *
     * @param iterator An iterator whose elements ought to be understood as holder type elements.
     * @param size The number of elements which ought to be read.
     * @param <T> The generic type parameter for the iterator.
     * @return An array of target types converted from the holder type elements read from the iterator.
     */
    <T> TargetArrayType readAndConvertForeignDataFrom( Iterator<T> iterator, int size );

    /**
     *  This method expects the provided stream to spit out bytes which can be read as target type elements.
     *  It then ought to convert these to an array of holder types of the specified size.
     *
     * @param stream A DataInput stream whose data ought to be read as target type elements.
     * @param size The number of elements which ought to be read.
     * @return An array of holder types converted from the target type elements read from the stream.
     * @throws IOException If reading from the stream was not successful.
     */
    HolderArrayType readForeignDataFrom (DataInput stream, int size ) throws IOException;

    /**
     *  This method expects the provided iterator to return elements which can be read as holder type elements.
     *  It then will write them into an array of holder types of the specified size.
     *
     * @param iterator An iterator whose elements ought to be understood as holder type elements.
     * @param size The number of elements which ought to be read and then written into an array.
     * @param <T> The generic type parameter for the iterator.
     * @return An array of holder types populated by holder elements read from the iterator.
     */
    <T> HolderArrayType readForeignDataFrom( Iterator<T> iterator, int size );

    /**
     *  This method writes all the target type elements returned by the provided iterator
     *  and write them into the provided "DataOutput" stream as bytes.
     *
     * @param stream The output stream which ought to be feed with bytes of the elements provided by the iterator.
     * @param iterator The iterator whose returned elements ought to be translated into bytes for the output stream.
     * @throws IOException An IOException forcing the caller to handle in case the conversion fails.
     */
    void writeDataTo( DataOutput stream, Iterator<TargetType> iterator ) throws IOException;

    /**
     *  This method is a generic converter from any object to an instance of the
     *  HolderType parameter specified by an implementation of this interface.
     *
     * @param from The object which ought to be converted to an instance of the holder type.
     * @return An instance of the holder type based on the conversion of the provided object.
     */
    HolderType convertToHolder( Object from );

    /**
     *  This method is a generic converter from any object to an instance of the
     *  HolderType parameter specified by an implementation of this interface.
     *
     * @param from The object which ought to be converted to an instance of the holder array type.
     * @return An instance of the holder array type based on the conversion of the provided object.
     */
    HolderArrayType convertToHolderArray( Object from );

    /**
     *  This method is a generic converter from any object to an instance of the
     *  TargetType parameter specified by an implementation of this interface.
     *
     * @param from The object which ought to be converted to an instance of the target type.
     * @return An instance of the target type based on the conversion of the provided object.
     */
    TargetType convertToTarget( Object from );

    /**
     *  This method is a generic converter from any object to an instance of the
     *  TargetArrayType parameter specified by an implementation of this interface.
     *
     * @param from The object which ought to be converted to an instance of the target array type.
     * @return An instance of the target array type based on the conversion of the provided object.
     */
    TargetArrayType convertToTargetArray( Object from );


}
