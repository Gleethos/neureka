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

package neureka.utility;

import neureka.dtype.DataType;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
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
public class DataConverter
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
     *  Besides Converter lambdas there are also Class objects used as keys.
     *  The keys of the outer Map represent "from types", whereas inner keys
     *  represent "to types".
     *  This allows for fast Converter access for a given type pair!
     */
    private static Map<Class<?>, Map<Class, Conversion>> _converters = new HashMap<>();

    /**
     *  This class is a singleton.
     *  Therefore it stores the following static "_instance" variable.
     */
    private static DataConverter _instance = new DataConverter();

    /**
     *  This method returns the singleton.
     *
     * @return The singleton instance of this class.
     */
    public static DataConverter instance() {
        return _instance;
    }

    /**
     *  This constructor is private because the DataConverter class is a singleton.
     *  Within the constructor the Converter lambdas are being set for a given
     *  "from"- and "to"- Class pair.
     */
    private DataConverter()
    {
        _set( byte[].class, float[].class, Utility::byteToFloat );
        _set( byte[].class, double[].class, Utility::byteToDouble );
        _set( byte[].class, short[].class, Utility::byteToShort );
        _set( byte[].class, int[].class, Utility::byteToInt );
        _set( byte[].class, long[].class, Utility::byteToLong );
        _set( byte[].class, BigInteger[].class, Utility::byteToBigInteger );

        _set( float[].class, double[].class, Utility::floatToDouble );
        _set( float[].class, int[].class,    Utility::floatToInt );
        _set( float[].class, short[].class, Utility::floatToShort );
        _set( float[].class, byte[].class, Utility::floatToByte );
        _set( float[].class, long[].class, Utility::floatToLong );
        _set( float[].class, BigInteger[].class, Utility::floatToBigInteger );

        _set( int[].class, float[].class,    Utility::intToFloat );
        _set( int[].class, double[].class,   Utility::intToDouble );
        _set( int[].class, long[].class, Utility::intToLong );
        _set( int[].class, short[].class, Utility::intToShort );
        _set( int[].class, BigInteger[].class, Utility::intToBigInteger );
        _set( int[].class, byte[].class, Utility::intToByte );

        _set( long[].class, byte[].class, Utility::longToByte );
        _set( long[].class, short[].class, Utility::longToShort );
        _set( long[].class, int[].class, Utility::longToInt );
        _set( long[].class, float[].class, Utility::longToFloat );
        _set( long[].class, double[].class, Utility::longToDouble );
        _set( long[].class, BigInteger[].class, Utility::longToBigInteger );

        _set( short[].class, long[].class, Utility::shortToLong );
        _set( short[].class, double[].class, Utility::shortToDouble );
        _set( short[].class, float[].class, Utility::shortToFloat );
        _set( short[].class, int[].class, Utility::shortToInt );
        _set( short[].class, byte[].class, Utility::shortToByte );
        _set( short[].class, BigInteger[].class, Utility::shortToBigInteger );

        _set( double[].class, byte[].class,  Utility::doubleToByte );
        _set( double[].class, short[].class, Utility::doubleToShort );
        _set( double[].class, int[].class, Utility::doubleToInt );
        _set( double[].class, BigInteger[].class, Utility::doubleToBigInteger );
        _set( double[].class, long[].class, Utility::doubleToLong );
        _set( double[].class, float[].class, Utility::doubleToFloat );

        _set( List.class, int[].class, thing -> thing.stream().mapToInt( i -> (int) i ).toArray() );
        _set( List.class, double[].class, thing -> thing.stream().mapToDouble( i -> (double) i ).toArray() );
        _set( List.class, long[].class, thing -> thing.stream().mapToLong( i -> (long) i ).toArray() );
        _set( BigInteger.class, Double.class, BigInteger::doubleValue );
        _set( BigDecimal.class, Double.class, BigDecimal::doubleValue );
        _set( Integer.class, Double.class, Integer::doubleValue );
        _set( Integer.class, Float.class, Integer::floatValue );
        _set( Integer.class, Short.class, Integer::shortValue );
        _set( Integer.class, Byte.class, Integer::byteValue );

        _set( Float[].class, float[].class,   Utility::objFloatsToPrimFloats );
        _set( Integer[].class, int[].class,   Utility::objIntsToPrimInts );
        _set( Long[].class, long[].class,     Utility::objLongsToPrimLongs );
        _set( Double[].class, double[].class, Utility::objDoublesToPrimDoubles );
        _set( Short[].class, short[].class,   Utility::objShortsToPrimShorts );
        _set( Byte[].class, byte[].class,     Utility::objBytesToPrimBytes );
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
        Map<Class, Conversion> fromSpecific = _converters.get( from.getClass() );
        if ( fromSpecific == null ) {
            String fromName = from.getClass().getSimpleName();
            String toName = to.getSimpleName();
            String message =
                    "Conversion from '"+fromName+"' to '"+toName+"' could not be performed.\n" +
                            "No converter lambdas were found for type '"+fromName+"'!";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        Map<Class, Conversion> tos = _converters.get(from.getClass());
        if ( tos == null ) {
            String message = "Conversion from type '"+from.getClass()+"' to '"+to+"' failed: No converters found for '"+from.getClass()+"'.";
            _LOG.error(message);
            throw new IllegalArgumentException(message);
        }
        Conversion<Object, Object> conversion = tos.get(to);
        if ( conversion == null ) {
            String message = "No converter found from type '"+from.getClass()+"' to '"+to+"'.";
            _LOG.error(message);
            throw new IllegalArgumentException(message);
        }
        return (T) conversion.go(from);
    }



    public <F extends Number, T> T convert( F[] from, Class<T> to, int size ) {
        if ( from.length == size )
            return convert( from, to );
        else {
            Iterator<F> stream = IntStream.iterate(0, i -> i + 1).limit(size).mapToObj(i -> from[i % from.length] ).iterator();
            int index = 0;
            if ( from instanceof Integer[] ) {
                Integer[] array = new Integer[size];
                for ( Iterator<F> it = stream; it.hasNext(); ) { array[ index ] = (Integer) it.next(); index++; }
                return convert( array, to );
            } else if ( from instanceof Double[] ) {
                Double[] array = new Double[size];
                for ( Iterator<F> it = stream; it.hasNext(); ) { array[ index ] = (Double) it.next(); index++; }
                return convert( array, to );
            } else if ( from instanceof Float[] ) {
                Float[] array = new Float[size];
                for ( Iterator<F> it = stream; it.hasNext(); ) { array[ index ] = (Float) it.next(); index++; }
                return convert( array, to );
            } else if ( from instanceof Short[] ) {
                Short[] array = new Short[size];
                for ( Iterator<F> it = stream; it.hasNext(); ) { array[ index ] = (Short) it.next(); index++; }
                return convert( array, to );
            } else if ( from instanceof Long[] ) {
                Long[] array = new Long[size];
                for ( Iterator<F> it = stream; it.hasNext(); ) { array[ index ] = (Long) it.next(); index++; }
                return convert( array, to );
            } else if ( from instanceof Byte[] ) {
                Byte[] array = new Byte[size];
                for ( Iterator<F> it = stream; it.hasNext(); ) { array[ index ] = (Byte) it.next(); index++; }
                return convert( array, to );
            }
            Object[] array = new Object[ size ];
            for ( Iterator<F> it = stream; it.hasNext(); ) { array[ index ] = it.next(); index++; }
            return convert( array, to );
        }
    }

    /**
     *  This is a static utility class containing the actual conversion logic
     *  which is usually referenced by the Converter lambdas via method signatures...
     *  Besides that it also provides the ability to create seeded arrays of data.
     */
    public static class Utility
    {
        public static float[] objFloatsToPrimFloats( Float[] objects ) {
            float[] array = new float[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        public static double[] objDoublesToPrimDoubles( Double[] objects ) {
            double[] array = new double[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        public static int[] objIntsToPrimInts( Integer[] objects ) {
            int[] array = new int[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        public static long[] objLongsToPrimLongs( Long[] objects ) {
            long[] array = new long[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        public static short[] objShortsToPrimShorts( Short[] objects ) {
            short[] array = new short[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        public static byte[] objBytesToPrimBytes( Byte[] objects ) {
            byte[] array = new byte[objects.length];
            for ( int i = 0; i < array.length; i++ ) {
                array[i] = objects[i];
            }
            return array;
        }

        public static double[] newSeededDoubleArray(String seed, int size) {
            return newSeededDoubleArray(_longStringHash(seed), size);
        }

        public static double[] newSeededDoubleArray(long seed, int size) {
            return seededDoubleArray(new double[size], seed);
        }

        public static double[] seededDoubleArray(double[] array, String seed) {
            return seededDoubleArray(array, _longStringHash(seed));
        }

        public static double[] seededDoubleArray(double[] array, long seed) {
            Random dice = new Random();
            dice.setSeed(seed);
            for( int i=0; i<array.length; i++ ) array[ i ] = dice.nextGaussian();
            return array;
        }

        public static float[] newSeededFloatArray(String seed, int size) {
            return newSeededFloatArray(_longStringHash(seed), size);
        }

        public static float[] newSeededFloatArray(long seed, int size) {
            return seededFloatArray(new float[size], seed);
        }

        public static float[] seededFloatArray(float[] array, String seed) {
            return seededFloatArray(array, _longStringHash(seed));
        }

        public static float[] seededFloatArray(float[] array, long seed) {
            Random dice = new Random();
            dice.setSeed(seed);
            for(int i=0; i<array.length; i++) array[ i ] = (float)dice.nextGaussian();
            return array;
        }

        private static long _longStringHash(String string)
        {
            long h = 1125899906842597L; // prime
            int len = string.length();
            for ( int i = 0; i < len; i++ ) h = 31*h + string.charAt( i );
            return h;
        }


        public static short[] byteToShort( byte[] data ) {
            if ( data == null ) return null;
            short[] newData = new short[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static BigInteger[] byteToBigInteger( byte[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = BigInteger.valueOf( data[ i ] );
            return newData;
        }

        public static float[] doubleToFloat( double[] data ) {
            if ( data == null ) return null;
            float[] newData = new float[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (float) data[ i ];
            return newData;
        }

        public static byte[] doubleToByte( double[] data ) {
            if ( data == null ) return null;
            byte[] newData = new byte[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        public static short[] doubleToShort( double[] data ) {
            if ( data == null ) return null;
            short[] newData = new short[ data.length ];
            for(int i=0; i<data.length; i++) newData[ i ] = (short) data[ i ];
            return newData;
        }

        public static long[] doubleToLong( double[] data ) {
            if ( data == null ) return null;
            long[] newData = new long[ data.length ];
            for(int i=0; i<data.length; i++) newData[ i ] = (long) data[ i ];
            return newData;
        }

        public static double[] floatToDouble(float[] data) {
            if (data==null) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (double)data[ i ];
            return newData;
        }

        public static byte[] floatToByte( float[] data ) {
            if (data==null) return null;
            byte[] newData = new byte[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        public static short[] floatToShort( float[] data ) {
            if (data==null) return null;
            short[] newData = new short[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (short) data[ i ];
            return newData;
        }

        public static long[] floatToLong( float[] data ) {
            if (data==null) return null;
            long[] newData = new long[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (long) data[ i ];
            return newData;
        }

        public static double[] shortToDouble(short[] data) {
            if (data==null) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static double[] byteToDouble(byte[] data) {
            if (data==null) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static float[] byteToFloat(byte[] data) {
            if (data==null) return null;
            float[] newData = new float[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static float[] shortToFloat(short[] data) {
            if (data==null) return null;
            float[] newData = new float[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static int[] byteToInt(byte[] data) {
            if (data==null) return null;
            int[] newData = new int[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static int[] shortToInt(short[] data) {
            if (data==null) return null;
            int[] newData = new int[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static byte[] shortToByte(short[] data) {
            if (data==null) return null;
            byte[] newData = new byte[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        public static long[] byteToLong(byte[] data) {
            if (data==null) return null;
            long[] newData = new long[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static long[] shortToLong(short[] data) {
            if (data==null) return null;
            long[] newData = new long[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static BigInteger[] shortToBigInteger(short[] data) {
            if (data==null) return null;
            BigInteger[] newData = new BigInteger[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = BigInteger.valueOf( data[ i ] );
            return newData;
        }

        public static float[] intToFloat(int[] data) {
            if ( data == null ) return null;
            float[] newData = new float[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (float) data[ i ];
            return newData;
        }

        public static int[] floatToInt(float[] data) {
            if ( data == null ) return null;
            int[] newData = new int[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (int) data[ i ];
            return newData;
        }

        public static BigInteger[] floatToBigInteger( float[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i < data.length; i++ ) newData[ i ] = BigInteger.valueOf( (int) data[i] );
            return newData;
        }

        public static int[] doubleToInt(double[] data) {
            if ( data == null ) return null;
            int[] newData = new int[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (int) data[ i ];
            return newData;
        }

        public static BigInteger[] doubleToBigInteger( double[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i < data.length; i++ ) newData[ i ] = BigInteger.valueOf( (long) data[i] );
            return newData;
        }

        public static double[] intToDouble(int[] data) {
            if ( data == null ) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static long[] intToLong( int[] data ) {
            if ( data == null ) return null;
            long[] newData = new long[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static short[] intToShort( int[] data ) {
            if ( data == null ) return null;
            short[] newData = new short[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (short) data[ i ];
            return newData;
        }

        public static byte[] intToByte( int[] data ) {
            if ( data == null ) return null;
            byte[] newData = new byte[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        public static BigInteger[] intToBigInteger( int[] data ) {
            if ( data == null ) return null;
            BigInteger[] newData = new BigInteger[ data.length ];
            for ( int i = 0; i < data.length; i++ ) newData[ i ] = BigInteger.valueOf( data[i] );
            return newData;
        }

        public static byte[] longToByte(long[] data) {
            if (data==null) return null;
            byte[] newData = new byte[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (byte) data[ i ];
            return newData;
        }

        public static short[] longToShort(long[] data) {
            if (data==null) return null;
            short[] newData = new short[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (short) data[ i ];
            return newData;
        }

        public static int[] longToInt(long[] data) {
            if (data==null) return null;
            int[] newData = new int[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (int) data[ i ];
            return newData;
        }

        public static float[] longToFloat(long[] data) {
            if (data==null) return null;
            float[] newData = new float[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (float) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static double[] longToDouble(long[] data) {
            if (data==null) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (double) data[ i ];
            return newData;
        }

        @Contract( pure = true )
        public static BigInteger[] longToBigInteger( long[] data ) {
            if (data==null) return null;
            BigInteger[] newData = new BigInteger[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = BigInteger.valueOf( data[ i ] );
            return newData;
        }

        //---

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
                if ( objects[ i % objects.length ] instanceof BigDecimal )
                    data[ i ] = ( (BigDecimal) objects[ i % objects.length ] ).floatValue();
                else if ( objects[ i % objects.length ] instanceof Integer )
                    data[ i ] = (Integer) objects[ i % objects.length ];
            }
            return data;
        }

        @Contract( pure = true )
        public static short[] objectsToShorts( Object[] objects, int targetSize ) {
            short[] data = new short[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof BigDecimal )
                    data[ i ] = ( (BigDecimal) objects[ i % objects.length ] ).shortValue();
                else if ( objects[ i % objects.length ] instanceof Integer )
                    data[ i ] = ( (Integer) objects[ i % objects.length ] ).shortValue();
            }
            return data;
        }

        @Contract( pure = true )
        public static byte[] objectsToBytes( Object[] objects, int targetSize ) {
            byte[] data = new byte[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof BigDecimal )
                    data[ i ] = ( (BigDecimal) objects[ i % objects.length ] ).byteValue();
                else if ( objects[ i % objects.length ] instanceof Integer )
                    data[ i ] = ( (Integer) objects[ i % objects.length ] ).byteValue();
            }
            return data;
        }

        @Contract( pure = true )
        public static long[] objectsToLongs( Object[] objects, int targetSize ) {
            long[] data = new long[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof BigDecimal )
                    data[ i ] = ( (BigDecimal) objects[ i % objects.length ] ).longValue();
                else if ( objects[ i % objects.length ] instanceof Integer )
                    data[ i ] = ( (Integer) objects[ i % objects.length ] ).longValue();
            }
            return data;
        }

        @Contract( pure = true )
        public static int[] objectsToInts( Object[] objects, int targetSize ) {
            int[] data = new int[ targetSize ];
            for ( int i = 0; i < data.length; i++ ) {
                if ( objects[ i % objects.length ] instanceof BigDecimal )
                    data[ i ] = ( (BigDecimal) objects[ i % objects.length ] ).shortValueExact();
                else if ( objects[ i % objects.length ] instanceof Integer )
                    data[ i ] = ( (Integer) objects[ i % objects.length ] );
            }
            return data;
        }
    }


}
