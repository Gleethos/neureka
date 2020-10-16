package neureka.utility;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class DataConverter
{
    private interface Conversion<FromType, ToType> {
        ToType go(FromType thing);
    }

    private static Map<Class, Map<Class, Conversion>> _converters = new HashMap<>();

    private static DataConverter _instance = new DataConverter();

    public static DataConverter instance(){
        return _instance;
    }

    private DataConverter(){
        set( float[].class, double[].class, Utility::floatToDouble);
        set( float[].class, int[].class,    Utility::floatToInt);
        set( double[].class, float[].class, Utility::doubleToFloat);
        set( int[].class, float[].class,    Utility::intToFloat);
        set( int[].class, double[].class,   Utility::intToDouble);
        set( List.class, int[].class, thing -> thing.stream().mapToInt(i-> (int) i).toArray() );
        set( List.class, double[].class, thing -> thing.stream().mapToDouble(i-> (double) i).toArray() );
        set( List.class, long[].class, thing -> thing.stream().mapToLong(i-> (long) i).toArray() );
        set( BigInteger.class, Double.class, BigInteger::doubleValue );
        set( BigDecimal.class, Double.class, BigDecimal::doubleValue );
        set( Integer.class, Double.class, Integer::doubleValue );
    }


    private <F,T> void set(
            Class<F> from, Class<T> to,
            Conversion<F,T> conversion
    ){
        Map<Class, Conversion> fromMap = _converters.get(from);
        if ( fromMap == null )
        {
            fromMap = new HashMap<>();
            fromMap.put(to, conversion);
            _converters.put( from, fromMap );
        } else {
            Conversion found = fromMap.get(to);
            if ( found != null ) throw new IllegalStateException("Conversion already present!");
            else fromMap.put(to, conversion);
        }
    }

    public <T> T convert(Object from, Class<T> to) {
        return (T) _converters.get(from.getClass()).get(to).go(from);
    }



    public static class Utility {
        public static double[] newSeededDoubleArray(String seed, int size){
            return newSeededDoubleArray(_longStringHash(seed), size);
        }

        public static double[] newSeededDoubleArray(long seed, int size){
            return seededDoubleArray(new double[size], seed);
        }

        public static double[] seededDoubleArray(double[] array, String seed){
            return seededDoubleArray(array, _longStringHash(seed));
        }

        public static double[] seededDoubleArray(double[] array, long seed){
            Random dice = new Random();
            dice.setSeed(seed);
            for(int i=0; i<array.length; i++) array[ i ] = dice.nextGaussian();
            return array;
        }

        public static float[] newSeededFloatArray(String seed, int size){
            return newSeededFloatArray(_longStringHash(seed), size);
        }

        public static float[] newSeededFloatArray(long seed, int size){
            return seededFloatArray(new float[size], seed);
        }

        public static float[] seededFloatArray(float[] array, String seed){
            return seededFloatArray(array, _longStringHash(seed));
        }

        public static float[] seededFloatArray(float[] array, long seed){
            Random dice = new Random();
            dice.setSeed(seed);
            for(int i=0; i<array.length; i++) array[ i ] = (float)dice.nextGaussian();
            return array;
        }

        private static long _longStringHash(String string)
        {
            long h = 1125899906842597L; // prime
            int len = string.length();
            for (int i = 0; i < len; i++) h = 31*h + string.charAt( i );
            return h;
        }

        public static float[] doubleToFloat(double[] data){
            if(data==null) return null;
            float[] newData = new float[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (float) data[ i ];
            return newData;
        }

        public static double[] floatToDouble(float[] data){
            if(data==null) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (double)data[ i ];
            return newData;
        }


        public static double[] shortToDouble(short[] data){
            if(data==null) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static double[] byteToDouble(byte[] data){
            if(data==null) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static float[] byteToFloat(byte[] data){
            if(data==null) return null;
            float[] newData = new float[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static float[] shortToFloat(short[] data){
            if(data==null) return null;
            float[] newData = new float[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static int[] byteToInt(byte[] data){
            if(data==null) return null;
            int[] newData = new int[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static int[] shortToInt(short[] data){
            if(data==null) return null;
            int[] newData = new int[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static long[] byteToLong(byte[] data){
            if(data==null) return null;
            long[] newData = new long[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

        public static long[] shortToLong(short[] data){
            if(data==null) return null;
            long[] newData = new long[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
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

        public static int[] doubleToInt(double[] data) {
            if ( data == null ) return null;
            int[] newData = new int[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = (int) data[ i ];
            return newData;
        }

        public static double[] intToDouble(int[] data) {
            if ( data == null ) return null;
            double[] newData = new double[data.length];
            for(int i=0; i<data.length; i++) newData[ i ] = data[ i ];
            return newData;
        }

    }


}
