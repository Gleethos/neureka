package neureka.utility;

import neureka.dtype.DataType;

import java.util.stream.IntStream;

public class ArrayUtils {

    public static Object optimizeArray( DataType<?> dataType, Object data, int size ) {
        if      ( data instanceof Integer[] ) return DataConverter.instance().convert( (Integer[]) data, int[].class,    size );
        else if ( data instanceof Double[]  ) return DataConverter.instance().convert( (Double[])  data, double[].class, size );
        else if ( data instanceof Float[]   ) return DataConverter.instance().convert( (Float[])   data, float[].class,  size );
        else if ( data instanceof Long[]    ) return DataConverter.instance().convert( (Long[])    data, long[].class,   size );
        else if ( data instanceof Short[]   ) return DataConverter.instance().convert( (Short[])   data, short[].class,  size );
        else if ( data instanceof Byte[]    ) return DataConverter.instance().convert( (Byte[])    data, byte[].class,   size );
        else if ( data instanceof Object[] )
            return ArrayUtils.optimizeObjectArray(dataType, (Object[]) data, size);
        else
            return data;
    }

    public static Object optimizeObjectArray( DataType<?> dataType, Object[] values, int size ) {
        Object data = values;
        IntStream indices = IntStream.iterate( 0, i -> i + 1 ).limit(size).map(i -> i % values.length );
        if      ( dataType == DataType.of(Double.class)  ) data = indices.mapToDouble( i -> (Double) values[i] ).toArray();
        else if ( dataType == DataType.of(Integer.class) ) data = indices.map( i -> (Integer) values[i] ).toArray();
        else if ( dataType == DataType.of(Long.class)    ) data = indices.mapToLong( i -> (Long) values[i] ).toArray();
        else if ( dataType == DataType.of(Float.class)   ) {
            float[] floats = new float[size];
            for( int i = 0; i < size; i++ ) floats[ i ] = (Float) values[ i % values.length ];
            data = floats;
        }
        else if ( dataType == DataType.of(Byte.class) ) {
            byte[] bytes = new byte[size];
            for( int i = 0; i < size; i++ ) bytes[ i ] = (Byte) values[ i % values.length ];
            data = bytes;
        }
        else if ( dataType == DataType.of(Short.class) ) {
            short[] shorts = new short[size];
            for( int i = 0; i < size; i++ ) shorts[ i ] = (Short) values[ i % values.length ];
            data = shorts;
        } else if ( values.length != size ) {
            Object[] objects = new Object[size];
            for( int i = 0; i < size; i++ ) objects[ i ] = values[ i % values.length ];
            data = objects;
        }
        return data;
    }


}
