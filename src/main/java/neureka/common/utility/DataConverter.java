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

   _____        _         _____                          _
  |  __ \      | |       / ____|                        | |
  | |  | | __ _| |_ __ _| |     ___  _ ____   _____ _ __| |_ ___ _ __
  | |  | |/ _` | __/ _` | |    / _ \| '_ \ \ / / _ \ '__| __/ _ \ '__|
  | |__| | (_| | || (_| | |___| (_) | | | \ V /  __/ |  | ||  __/ |
  |_____/ \__,_|\__\__,_|\_____\___/|_| |_|\_/ \___|_|   \__\___|_|

    A helpful little singleton class used for data conversion.

*/

package neureka.common.utility;

import neureka.Tsr;
import neureka.dtype.DataType;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 *  This class is a singleton.
 *  Its sole job is to simply take in any kind ob object and convert it into
 *  another object of a provided Class type...
 *  In essence the {@link DataConverter} is merely a utility class.
 *  It also contains a nested static class named {@link Utility} which
 *  provides useful methods to handle primitive data types and arrays
 *  of said types.
 */
public final class DataConverter
{
    private final static Logger _LOG = LoggerFactory.getLogger(DataConverter.class);

    /**
     *  This interface declares a simple lambda which represents type conversion implementations...
     *  These conversion lambdas are then stored within a nested Map that can be extended easily.
     *  The structure of this interface is not different to the java.util.function.Function<T,R>
     *  interface, however for the sake of descriptiveness and completeness the interface is
     *  still redefined and named accordingly!
     *
     * @param <FromType> The type that is being passed to the lambda.
     * @param <ToType> The target type that is returned.
     */
    private interface Conversion<FromType, ToType> { ToType go(FromType thing); }

    /**
     *  This nested Map field manages Converter lambda instances!
     *  Other than converter lambdas, there are also Class objects used as keys.
     *  The keys of the outer Map represent "from types", whereas inner keys
     *  represent "to types".
     *  This allows for fast Converter access for a given type pair!
     */
    private static Map<Class<?>, Map<Class, Conversion>> _converters = new HashMap<>(128);

    /**
     *  This class is a singleton.
     *  Meaning it stores the following static "_instance" variable.
     */
    private static final DataConverter _instance = new DataConverter();

    /**
     *  This method returns the singleton.
     *
     * @return The singleton instance of this class.
     */
    public static DataConverter instance() { return _instance; }

    /**
     *  This constructor is private because the DataConverter class is a singleton.
     *  Within the constructor the Converter lambdas are being set for a given
     *  "from"- and "to"- Class pair.
     */
    private DataConverter()
    {
        _set( byte[].class, float[].class,      Utility::byteToFloat );
        _set( byte[].class, double[].class,     Utility::byteToDouble );
        _set( byte[].class, short[].class,      Utility::byteToShort );
        _set( byte[].class, int[].class,        Utility::byteToInt );
        _set( byte[].class, long[].class,       Utility::byteToLong );
        _set( byte[].class, BigInteger[].class, Utility::byteToBigInteger );

        _set( float[].class, double[].class,     Utility::floatToDouble );
        _set( float[].class, int[].class,        Utility::floatToInt );
        _set( float[].class, short[].class,      Utility::floatToShort );
        _set( float[].class, byte[].class,       Utility::floatToByte );
        _set( float[].class, long[].class,       Utility::floatToLong );
        _set( float[].class, BigInteger[].class, Utility::floatToBigInteger );

        _set( int[].class, float[].class,      Utility::intToFloat );
        _set( int[].class, double[].class,     Utility::intToDouble );
        _set( int[].class, long[].class,       Utility::intToLong );
        _set( int[].class, short[].class,      Utility::intToShort );
        _set( int[].class, BigInteger[].class, Utility::intToBigInteger );
        _set( int[].class, byte[].class,       Utility::intToByte );

        _set( long[].class, byte[].class,       Utility::longToByte );
        _set( long[].class, short[].class,      Utility::longToShort );
        _set( long[].class, int[].class,        Utility::longToInt );
        _set( long[].class, float[].class,      Utility::longToFloat );
        _set( long[].class, double[].class,     Utility::longToDouble );
        _set( long[].class, BigInteger[].class, Utility::longToBigInteger );

        _set( short[].class, long[].class,       Utility::shortToLong );
        _set( short[].class, double[].class,     Utility::shortToDouble );
        _set( short[].class, float[].class,      Utility::shortToFloat );
        _set( short[].class, int[].class,        Utility::shortToInt );
        _set( short[].class, byte[].class,       Utility::shortToByte );
        _set( short[].class, BigInteger[].class, Utility::shortToBigInteger );

        _set( double[].class, byte[].class,       Utility::doubleToByte );
        _set( double[].class, short[].class,      Utility::doubleToShort );
        _set( double[].class, int[].class,        Utility::doubleToInt );
        _set( double[].class, BigInteger[].class, Utility::doubleToBigInteger );
        _set( double[].class, long[].class,       Utility::doubleToLong );
        _set( double[].class, float[].class,      Utility::doubleToFloat );
        _set( double[].class, boolean[].class,    Utility::doubleToBool );

        _set( boolean[].class, double[].class,    Utility::boolToDouble );
        _set( boolean[].class, float[].class,     Utility::boolToFloat );

        _set( List.class, int[].class,    thing -> thing.stream().mapToInt(    i -> convert(i, Integer.class) ).toArray() );
        _set( List.class, double[].class, thing -> thing.stream().mapToDouble( i -> convert(i, Double.class) ).toArray() );
        _set( List.class, long[].class,   thing -> thing.stream().mapToLong(   i -> convert(i, Long.class) ).toArray() );

        _set( BigInteger.class, Double.class, BigInteger::doubleValue );
        _set( BigInteger.class, Float.class, BigInteger::floatValue );
        _set( BigInteger.class, Long.class, BigInteger::longValue );
        _set( BigInteger.class, Integer.class, BigInteger::intValue );
        _set( BigInteger.class, Short.class, BigInteger::shortValue );
        _set( BigInteger.class, Byte.class, BigInteger::byteValue );

        _set( BigDecimal.class, Double.class, BigDecimal::doubleValue );
        _set( BigDecimal.class, Float.class, BigDecimal::floatValue );
        _set( BigDecimal.class, Long.class, BigDecimal::longValue );
        _set( BigDecimal.class, Integer.class, BigDecimal::intValue );
        _set( BigDecimal.class, Short.class, BigDecimal::shortValue );
        _set( BigDecimal.class, Byte.class, BigDecimal::byteValue );

        _set( Integer.class, Double.class, Integer::doubleValue );
        _set( Integer.class, Float.class,  Integer::floatValue );
        _set( Integer.class, Byte.class,   Integer::byteValue );
        _set( Integer.class, Short.class,  Integer::shortValue );
        _set( Integer.class, Long.class,   Integer::longValue );

        _set( Double.class, Float.class,   Double::floatValue );
        _set( Double.class, Integer.class, Double::intValue );
        _set( Double.class, Byte.class,    Double::byteValue );
        _set( Double.class, Short.class,   Double::shortValue );
        _set( Double.class, Long.class,    Double::longValue );

        _set( Float.class, Double.class,  Float::doubleValue );
        _set( Float.class, Integer.class, Float::intValue );
        _set( Float.class, Byte.class,    Float::byteValue );
        _set( Float.class, Short.class,   Float::shortValue );
        _set( Float.class, Long.class,    Float::longValue );

        _set( Boolean.class, Double.class,   b -> b ? 1d : 0d );
        _set( Boolean.class, Float.class,    b -> b ? 1f : 0f );
        _set( Boolean.class, Integer.class,  b -> b ? 1 : 0   );
        _set( Boolean.class, Long.class,     b -> b ? 1L : 0L );
        _set( Boolean.class, Byte.class,     b -> b ? (byte) 1 : (byte) 0 );

        _set( Float[].class,     float[].class,   Utility::objFloatsToPrimFloats );
        _set( Integer[].class,   int[].class,     Utility::objIntsToPrimInts );
        _set( Long[].class,      long[].class,    Utility::objLongsToPrimLongs );
        _set( Double[].class,    double[].class,  Utility::objDoublesToPrimDoubles );
        _set( Short[].class,     short[].class,   Utility::objShortsToPrimShorts );
        _set( Byte[].class,      byte[].class,    Utility::objBytesToPrimBytes );
        _set( Boolean[].class,   boolean[].class, Utility::objBooleansToPrimBooleans );
        _set( Character[].class, char[].class,    Utility::objCharsToPrimChars );

        _set( Object[].class, int[].class, data -> {
            return
               Utility.intStream( 1_000, data.length )
                   .map( i -> {
                        if ( data[ i ] instanceof Number )
                            return ( (Number) data[ i ] ).intValue();
                        else if ( data[ i ] instanceof String )
                            return (int) Double.parseDouble( (String) data[ i ] );
                        else
                            throw new IllegalArgumentException(
                                "Cannot convert type '"+data[ i ].getClass().getSimpleName()+"' to int!"
                            );
                    })
                    .toArray();
        });
    }

    /**
     *  This method fills the previously defined nested Map field of this class
     *  with Converter instances using the provided Class objects as keys.
     *
     * @param from The Class of the type that is being put into the given converter.
     * @param to The Class of the type that is being returned by the given converter.
     * @param conversion The Converter lambda instance, namely : the conversion for the provided types.
     * @param <F> The "from" type argument.
     * @param <T> The "to" type argument.
     */
    private <F,T> void _set(
            Class<F> from, Class<T> to,
            Conversion<F,T> conversion
    ) {
        Map<Class, Conversion> fromMap = _converters.get(from);
        if ( fromMap == null )
        {
            fromMap = new HashMap<>();
            fromMap.put(to, conversion);
            fromMap.put(DataType.of(to).getTypeClass(), conversion);
            _converters.put( from, fromMap );
        } else {
            Conversion found = fromMap.get(to);
            if ( found != null ) throw new IllegalStateException(
                    "Conversion already present! From class '"+from.getName()+"'. To clas '"+to.getName()+"'."
                    );
            else fromMap.put(to, conversion);
        }
    }

    /**
     *  This method embodies the purpose of this class.
     *  It receives objects for type conversion and queries the request
     *  through the nested "_converters" Map instance.
     *
     * @param from The object which ought to be converted.
     * @param to The target type for the provided object.
     * @param <T> The type parameter of the "to" Class.
     * @return The target object created by a Converter lambda.
     */
    public <T> T convert( Object from, Class<T> to ) {
        if ( from == null ) return null;
        if ( from.getClass() == to ) return (T) from;
        if ( to == String.class ) return (T) from.toString();

        Map<Class, Conversion> fromSpecific = _converters.get( from.getClass() );
        if ( fromSpecific == null ) {
            for ( Class<?> fromClass : _converters.keySet() )
                if ( fromClass.isAssignableFrom(from.getClass()) ) {
                    fromSpecific =  _converters.get( fromClass );
                    break;
                }
        }
        if ( fromSpecific == null ) {
            String fromName = from.getClass().getSimpleName();
            String toName = to.getSimpleName();
            String message =
                    "Conversion from '"+fromName+"' to '"+toName+"' could not be performed.\n" +
                            "No converter lambdas were found for type '"+fromName+"'!";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        Conversion<Object, Object> conversion = fromSpecific.get(to);
        if ( conversion == null ) {
            String message = "No converter found from type '"+from.getClass()+"' to '"+to+"'.";
            _LOG.error(message);
            throw new IllegalArgumentException(message);
        }
        return (T) conversion.go(from);
    }

    /**
     *  This is a stateful and parallelized converter for converting the internal data array of a tensor
     *  to another data array based on a provided lambda.
     *  The converter will consider tensors with more complex access pattern like
     *  for example those of slices.
     */
    public static class ForTensor {

        private final NDConfiguration.IndexToIndexFunction _access;
        private final int _size;

        public ForTensor( Tsr<?> t ) {
            _access = t.getNDConf().getIndexToIndexAccessPattern();
            _size = t.size();
        }

        public float[] toFloatArray( Function<Integer, Number> source ) {
            float[] data = new float[ _size ];
            Utility.intStream( 1_500, _size )
                    .forEach( i -> data[i] = source.apply(_access.map(i)).floatValue());
            return data;
        }

        public byte[] toByteArray( Function<Integer, Number> source ) {
            byte[] data = new byte[ _size ];
            Utility.intStream( 1_500, _size )
                    .forEach( i -> data[i] = source.apply(_access.map(i)).byteValue() );
            return data;
        }

        public long[] toLongArray( Function<Integer, Number> source ) {
            long[] data = new long[ _size ];
            Utility.intStream( 1_500, _size )
                    .forEach( i -> data[i] = source.apply(_access.map(i)).longValue());
            return data;
        }

        public int[] toIntArray( Function<Integer, Number> source ) {
            int[] data = new int[ _size ];
            Utility.intStream( 1_500, _size )
                    .forEach( i -> data[i] = source.apply(_access.map(i)).intValue());
            return data;
        }

        public double[] toDoubleArray( Function<Integer, Number> source ) {
            double[] data = new double[ _size ];
            Utility.intStream( 1_500, _size )
                    .forEach( i -> data[i] = source.apply(_access.map(i)).doubleValue() );
            return data;
        }

        public short[] toShortArray( Function<Integer, Number> source ) {
            short[] data = new short[ _size ];
            Utility.intStream( 1_500, _size )
                    .forEach( i -> data[i] = source.apply(_access.map(i)).shortValue() );
            return data;
        }

        public Object[] toObjectArray( Function<Integer, Object> source ) {
            Object[] data = new Object[ _size ];
            Utility.intStream( 1_500, _size )
                    .forEach( i -> data[i] = source.apply(_access.map(i)) );
            return data;
        }

    }

    /**
     *  This is a static utility class containing the actual conversion logic
     *  which is usually referenced by the Converter lambdas via method signatures...
     *  Other than that it also provides the ability to create seeded arrays of data.
     */
    public static class Utility
    {
        @Contract( pure = true )
        public static float[] objFloatsToPrimFloats( Float[] objects ) {
            float[] array = new float[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        @Contract( pure = true )
        public static double[] objDoublesToPrimDoubles( Double[] objects ) {
            double[] array = new double[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        @Contract( pure = true )
        public static int[] objIntsToPrimInts( Integer[] objects ) {
            int[] array = new int[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        @Contract( pure = true )
        public static long[] objLongsToPrimLongs( Long[] objects ) {
            long[] array = new long[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        @Contract( pure = true )
        public static short[] objShortsToPrimShorts( Short[] objects ) {
            short[] array = new short[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        @Contract( pure = true )
        public static byte[] objBytesToPrimBytes( Byte[] objects ) {
            byte[] array = new byte[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        @Contract( pure = true )
        public static boolean[] objBooleansToPrimBooleans( Boolean[] objects ) {
            boolean[] array = new boolean[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        @Contract( pure = true )
        public static char[] objCharsToPrimChars( Character[] objects ) {
            char[] array = new char[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        @Contract( pure = true )
        public static double[] newSeededDoubleArray( String seed, int size ) {
            return newSeededDoubleArray( _longStringHash( seed ), size );
        }

        @Contract( pure = true )
        public static double[] newSeededDoubleArray( long seed, int size ) {
            return seededDoubleArray( new double[size], seed );
        }

        @Contract( pure = true )
        public static double[] seededDoubleArray( double[] array, String seed ) {
            return seededDoubleArray(array, _longStringHash( seed ));
        }

        @Contract( pure = true )
        public static double[] seededDoubleArray( double[] array, long seed ) {
            Random dice = new Random();
            dice.setSeed( seed );
            for ( int i = 0; i < array.length; i++ ) array[ i ] = dice.nextGaussian();
            return array;
        }

        @Contract( pure = true )
        public static float[] newSeededFloatArray( String seed, int size ) {
            return newSeededFloatArray( _longStringHash( seed ), size );
        }

        @Contract( pure = true )
        public static float[] newSeededFloatArray( long seed, int size ) {
            return seededFloatArray( new float[size], seed );
        }

        @Contract( pure = true )
        public static float[] seededFloatArray( float[] array, String seed ) {
            return seededFloatArray( array, _longStringHash( seed ) );
        }

        @Contract( pure = true )
        public static float[] seededFloatArray( float[] array, long seed ) {
            Random dice = new Random();
            dice.setSeed( seed );
            for ( int i = 0; i < array.length; i++ ) array[ i ] = (float) dice.nextGaussian();
            return array;
        }

        @Contract( pure = true )
        public static int[] seededIntArray( int[] array, String seed ) {
            return seededIntArray(array, _longStringHash( seed ));
        }

        @Contract( pure = true )
        public static int[] seededIntArray( int[] array, long seed ) {
            Random dice = new Random();
            dice.setSeed( seed );
            for ( int i = 0; i < array.length; i++ ) array[ i ] = dice.nextInt();
            return array;
        }

        @Contract( pure = true )
        public static short[] seededShortArray( short[] array, String seed ) {
            return seededShortArray(array, _longStringHash( seed ));
        }

        @Contract( pure = true )
        public static short[] seededShortArray( short[] array, long seed ) {
            Random dice = new Random();
            dice.setSeed( seed );
            for ( int i = 0; i < array.length; i++ ) array[ i ] = (short) dice.nextInt();
            return array;
        }

        @Contract( pure = true )
        public static byte[] seededByteArray( byte[] array, String seed ) {
            return seededByteArray(array, _longStringHash( seed ));
        }

        @Contract( pure = true )
        public static byte[] seededByteArray( byte[] array, long seed ) {
            Random dice = new Random();
            dice.setSeed( seed );
            for ( int i = 0; i < array.length; i++ ) array[ i ] = (byte) dice.nextInt();
            return array;
        }

        @Contract( pure = true )
        public static long[] seededLongArray( long[] array, String seed ) {
            return seededLongArray( array, _longStringHash( seed ) );
        }

        @Contract( pure = true )
        public static long[] seededLongArray( long[] array, long seed ) {
            Random dice = new Random();
            dice.setSeed( seed );
            for ( int i = 0; i < array.length; i++ ) array[ i ] = dice.nextLong();
            return array;
        }

        @Contract( pure = true )
        public static boolean[] seededBooleanArray( boolean[] array, String seed ) {
            return seededBooleanArray(array, _longStringHash( seed ));
        }

        @Contract( pure = true )
        public static boolean[] seededBooleanArray( boolean[] array, long seed ) {
            Random dice = new Random();
            dice.setSeed( seed );
            for ( int i = 0; i < array.length; i++ ) array[ i ] = dice.nextBoolean();
            return array;
        }

        @Contract( pure = true )
        public static char[] seededCharacterArray( char[] array, String seed) {
            return seededCharacterArray(array, _longStringHash( seed ));
        }

        @Contract( pure = true )
        public static char[] seededCharacterArray( char[] array, long seed ) {
            Random dice = new Random();
            dice.setSeed( seed );
            for ( int i = 0; i < array.length; i++ ) array[ i ] = (char) dice.nextInt();
            return array;
        }

        @Contract( pure = true )
        private static long _longStringHash( String string )
        {
            long h = 1125899906842597L; // prime
            int len = string.length();
            for ( int i = 0; i < len; i++ ) h = 31 * h + string.charAt( i );
            return h;
        }

        @Contract( pure = true )
        public static short[] byteToShort( byte[] data ) {
            if ( data == null ) return null;
            short[] newData = new short[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static BigInteger[] byteToBigInteger( byte[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = BigInteger.valueOf( data[ i ] );
            return newData;
        }

        @Contract( pure = true )
        public static float[] doubleToFloat( double[] data ) {
            if ( data == null ) return null;
            float[] newData = new float[ data.length ];
            if ( data.length < 150_000 ) // Only very large arrays can be parallelized in this case!
                for ( int i = 0; i < data.length; i++) newData[ i ] = (float) data[ i ];
            else
                IntStream.range(0, data.length)
                        .parallel()
                        .forEach( i -> newData[ i ] = (float) data[ i ] );

            return newData;
        }

        @Contract( pure = true )
        public static byte[] doubleToByte( double[] data ) {
            if ( data == null ) return null;
            byte[] newData = new byte[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static short[] doubleToShort( double[] data ) {
            if ( data == null ) return null;
            short[] newData = new short[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = (short) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static long[] doubleToLong( double[] data ) {
            if ( data == null ) return null;
            long[] newData = new long[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = (long) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static boolean[] doubleToBool( double[] data ) {
            if ( data == null ) return null;
            boolean[] newData = new boolean[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = Math.floor(data[ i ]) >= 1;
            return newData;
        }

        @Contract( pure = true )
        public static double[] boolToDouble( boolean[] data ) {
            if ( data == null ) return null;
            double[] newData = new double[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = data[ i ] ? 1 : 0;
            return newData;
        }

        @Contract( pure = true )
        public static float[] boolToFloat( boolean[] data ) {
            if ( data == null ) return null;
            float[] newData = new float[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = data[ i ] ? 1 : 0;
            return newData;
        }

        @Contract( pure = true )
        public static double[] floatToDouble(float[] data) {
            if ( data == null ) return null;
            double[] newData = new double[ data.length ];
            if ( data.length < 150_000 ) // Only very large arrays can be parallelized in this case!
                for ( int i = 0; i < data.length; i++) newData[ i ] = data[ i ];
            else
                IntStream.range(0, data.length)
                        .parallel()
                        .forEach( i -> newData[ i ] = data[ i ] );

            return newData;
        }

        @Contract( pure = true )
        public static byte[] floatToByte( float[] data ) {
            if ( data == null ) return null;
            byte[] newData = new byte[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static short[] floatToShort( float[] data ) {
            if ( data == null ) return null;
            short[] newData = new short[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = (short) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static long[] floatToLong( float[] data ) {
            if ( data == null ) return null;
            long[] newData = new long[ data.length ];
            for ( int i = 0; i < data.length; i++) newData[ i ] = (long) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static double[] shortToDouble( short[] data ) {
            if ( data == null ) return null;
            double[] newData = new double[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static double[] byteToDouble( byte[] data ) {
            if ( data == null ) return null;
            double[] newData = new double[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static float[] byteToFloat( byte[] data ) {
            if ( data == null ) return null;
            float[] newData = new float[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static float[] shortToFloat( short[] data ) {
            if ( data == null ) return null;
            float[] newData = new float[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static int[] byteToInt( byte[] data ) {
            if ( data == null ) return null;
            int[] newData = new int[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static int[] shortToInt( short[] data ) {
            if ( data == null ) return null;
            int[] newData = new int[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static byte[] shortToByte(short[] data) {
            if ( data == null ) return null;
            byte[] newData = new byte[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static long[] byteToLong(byte[] data) {
            if ( data == null ) return null;
            long[] newData = new long[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static long[] shortToLong(short[] data) {
            if ( data == null ) return null;
            long[] newData = new long[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static BigInteger[] shortToBigInteger(short[] data) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = BigInteger.valueOf( data[ i ] );
            return newData;
        }

        @Contract( pure = true )
        public static float[] intToFloat(int[] data) {
            if ( data == null ) return null;
            float[] newData = new float[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (float) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static int[] floatToInt(float[] data) {
            if ( data == null ) return null;
            int[] newData = new int[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (int) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static BigInteger[] floatToBigInteger( float[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i < data.length; i++ ) newData[ i ] = BigInteger.valueOf( (int) data[i] );
            return newData;
        }

        @Contract( pure = true )
        public static int[] doubleToInt(double[] data) {
            if ( data == null ) return null;
            int[] newData = new int[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (int) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static BigInteger[] doubleToBigInteger( double[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i < data.length; i++ ) newData[ i ] = BigInteger.valueOf( (long) data[i] );
            return newData;
        }

        @Contract( pure = true )
        public static double[] intToDouble(int[] data) {
            if ( data == null ) return null;
            double[] newData = new double[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static long[] intToLong( int[] data ) {
            if ( data == null ) return null;
            long[] newData = new long[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static short[] intToShort( int[] data ) {
            if ( data == null ) return null;
            short[] newData = new short[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (short) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static byte[] intToByte( int[] data ) {
            if ( data == null ) return null;
            byte[] newData = new byte[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static BigInteger[] intToBigInteger( int[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i < data.length; i++ ) newData[ i ] = BigInteger.valueOf( data[i] );
            return newData;
        }

        @Contract( pure = true )
        public static byte[] longToByte(long[] data) {
            if ( data == null ) return null;
            byte[] newData = new byte[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static short[] longToShort(long[] data) {
            if ( data == null ) return null;
            short[] newData = new short[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (short) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static int[] longToInt(long[] data) {
            if ( data == null ) return null;
            int[] newData = new int[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (int) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static float[] longToFloat(long[] data) {
            if ( data == null ) return null;
            float[] newData = new float[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (float) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static double[] longToDouble(long[] data) {
            if ( data == null ) return null;
            double[] newData = new double[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = (double) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static BigInteger[] longToBigInteger( long[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i <data.length; i++) newData[ i ] = BigInteger.valueOf( data[ i ] );
            return newData;
        }

        @Contract( pure = true )
        public static double[] objectsToDoubles( Object[] objects, int targetSize ) {
            double[] data = new double[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof BigDecimal )
                    data[ i ] = ( (BigDecimal) objects[ i % objects.length ] ).doubleValue();
                else if ( objects[ i % objects.length ] instanceof Integer )
                    data[ i ] = (Integer) objects[ i % objects.length ];
            }
            return data;
        }

        @Contract( pure = true )
        public static float[] objectsToFloats( Object[] objects, int targetSize ) {
            float[] data = new float[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof Number )
                    data[ i ] = ( (Number) objects[ i % objects.length ] ).floatValue();
            }
            return data;
        }

        @Contract( pure = true )
        public static short[] objectsToShorts( Object[] objects, int targetSize ) {
            short[] data = new short[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof Number )
                    data[ i ] = ( (Number) objects[ i % objects.length ] ).shortValue();
            }
            return data;
        }

        @Contract( pure = true )
        public static byte[] objectsToBytes( Object[] objects, int targetSize ) {
            byte[] data = new byte[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof Number )
                    data[ i ] = ( (Number) objects[ i % objects.length ] ).byteValue();
            }
            return data;
        }

        @Contract( pure = true )
        public static long[] objectsToLongs( Object[] objects, int targetSize ) {
            long[] data = new long[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof Number )
                    data[ i ] = ( (Number) objects[ i % objects.length ] ).longValue();
            }
            return data;
        }

        @Contract( pure = true )
        public static int[] objectsToInts( Object[] objects, int targetSize ) {
            int[] data = new int[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof Number )
                    data[ i ] = ( (Number) objects[ i % objects.length ] ).intValue();
            }
            return data;
        }

        /**
         *  Use this to create a range based {@link IntStream}
         *  which is only parallel if the provided threshold
         *  smaller than the provided workload size.
         *
         * @param parallelThreshold If the {@code workload} is larger than the threshold then the returned stream will be parallel.
         * @param workload The number of integers processed by the returned stream.
         * @return A sequential or parallel {@link IntStream}.
         */
        public static IntStream intStream( int parallelThreshold, int workload ) {
            IntStream stream = IntStream.range( 0, workload );
            if ( workload >= parallelThreshold ) return stream.parallel();
            else return stream;
        }

    }


}
