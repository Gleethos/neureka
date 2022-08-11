package neureka;

import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.dtype.DataType;
import neureka.dtype.custom.*;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 *  The {@link TsrConstructor} is an <b>internal API</b> for receiving a wide range
 *  of different inputs and using them to populate the fields
 *  of freshly instantiated {@link neureka.Tsr} instances.
 *  The existence of this class is a symptom of the fact that there
 *  is a very large API for creating tensors in Neureka.
 *  This means that all the code dealing with verifying and converting
 *  API input (provided by various {@link neureka.Tsr#of} methods)
 *  sits inside this class instead of polluting the already very large
 *  {@link neureka.Tsr} class.
 */
final class TsrConstructor
{
    /**
     *  An interface defining methods for configuring a {@link neureka.Tsr}
     *  in the making...
     */
    public interface API {
        void   setType( DataType<?> type );
        void   setConf( NDConfiguration conf );
        void   setData( Object o );
        void   allocate( int size );
        Object getData();
        void   setIsVirtual(  boolean isVirtual );
    }

    private final API _API;

    public TsrConstructor(API API ) { _API = API; }

    /**
     *  This method is responsible for instantiating and setting the _conf variable.
     *  The core requirement for instantiating {@link NDConfiguration} interface implementation s
     *  is a shape array of integers which is being passed to the method... <br>
     *  <br>
     *
     * @param ndConstructor A producer of the {@link NDConfiguration} interface implementation.
     * @param makeVirtual A flag determining if the tensor should be actual or virtual (not fully allocated).
     * @param autoAllocate Determines if the underlying data array should be allocated or not.
     */
    public void configureFromNewShape(NDConstructor ndConstructor, boolean makeVirtual, boolean autoAllocate )
    { 
        _API.setIsVirtual( makeVirtual ); 
        if ( _API.getData() == null && autoAllocate ) _API.allocate( makeVirtual ? 1 : ndConstructor.getSize() );
        _API.setConf( ndConstructor.produceNDC( makeVirtual ) );
    }

    public void tryConstructing(
        NDConstructor ndConstructor,
        DataType<?> dataType,
        Object data,
        boolean trusted
    ) {
        LogUtil.nullArgCheck( ndConstructor, "ndConstructor", NDConstructor.class );
        LogUtil.nullArgCheck( ndConstructor.getShape(), "shape", int[].class );
        LogUtil.nullArgCheck( dataType, "dataType", DataType.class );
        if ( trusted ) {
            _API.setType( dataType );
            _API.setData( data );
            _API.setConf( ndConstructor.produceNDC( false ) );
            return;
        }
        LogUtil.nullArgCheck( data, "data", Object.class );

        int size = ndConstructor.getSize();
        if ( data instanceof List<?> ) {
            List<?> range = (List<?>) data;
            data = range.toArray();// TODO: This is probably wrong!
        }
        if ( data instanceof Object[] )
            data = _autoConvertAndOptimizeObjectArray( (Object[]) data, dataType, size );

        boolean isDefinitelyScalarValue = ( dataType == DataType.of( data.getClass() ) );

        if ( data instanceof Number && !isDefinitelyScalarValue ) {
            data = DataConverter.get().convert( data, dataType.getItemTypeClass() );
            isDefinitelyScalarValue = true;
        }

        if ( isDefinitelyScalarValue ) // This means that "data" is a single value!
            if ( constructAllFromOne(ndConstructor, data ) ) return;

        if (      data instanceof double[]  ) _constructForDoubles(ndConstructor, (double[]) data );
        else if ( data instanceof float[]   ) _constructForFloats(ndConstructor, (float[]) data );
        else if ( data instanceof int[]     ) _constructForInts(ndConstructor, (int[]) data );
        else if ( data instanceof byte[]    ) _constructForBytes(ndConstructor, (byte[]) data );
        else if ( data instanceof short[]   ) _constructForShorts(ndConstructor, (short[]) data );
        else if ( data instanceof boolean[] ) _constructForBooleans(ndConstructor, (boolean[]) data );
        else if ( data instanceof long[]    ) _constructForLongs(ndConstructor, (long[]) data );
        else {
            _API.setType(dataType);
            configureFromNewShape(ndConstructor, false, false);
            _API.setData(data);
        }
    }

    private Object _autoConvertAndOptimizeObjectArray( Object[] data, DataType<?> dataType, int size ) {
        if ( Arrays.stream( data ).anyMatch( e -> DataType.of(e.getClass()) != dataType ) )
            for ( int i = 0; i < ( data ).length; i++ )
                ( data )[i] = DataConverter.get().convert( ( (Object[]) data )[i], dataType.getItemTypeClass() );

        return _optimizeObjectArray(dataType, data, size);
    }

    public boolean constructAllFromOne( NDConstructor ndConstructor, Object data ) {
        if ( data instanceof Double    ) { _constructAllF64( ndConstructor,  (Double)    data ); return true; }
        if ( data instanceof Float     ) { _constructAllF32( ndConstructor,  (Float)     data ); return true; }
        if ( data instanceof Integer   ) { _constructAllI32( ndConstructor,  (Integer)   data ); return true; }
        if ( data instanceof Short     ) { _constructAllI16( ndConstructor,  (Short)     data ); return true; }
        if ( data instanceof Byte      ) { _constructAllI8( ndConstructor,   (Byte)      data ); return true; }
        if ( data instanceof Long      ) { _constructAllI64( ndConstructor,  (Long)      data ); return true; }
        if ( data instanceof Boolean   ) { _constructAllBool( ndConstructor, (Boolean)   data ); return true; }
        if ( data instanceof Character ) { _constructAllChar( ndConstructor, (Character) data ); return true; }
        if ( Number.class.isAssignableFrom( data.getClass() ) ) {
            _constructAllF64( ndConstructor, ((Number)data).doubleValue() ); return true;
        } else if ( !data.getClass().isArray() ) {
            _constructAll( ndConstructor, data ); return true;
        }
        return false;
    }

    private void _constructAllF64( NDConstructor ndConstructor, double value ) {
        _constructAll(ndConstructor, F64.class );
        ( (double[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllF32( NDConstructor ndConstructor, float value ) {
        _constructAll(ndConstructor, F32.class );
        ( (float[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI32( NDConstructor ndConstructor, int value ) {
        _constructAll(ndConstructor, I32.class );
        ( (int[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI16( NDConstructor ndConstructor, short value ) {
        _constructAll(ndConstructor, I16.class );
        ( (short[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI8( NDConstructor ndConstructor, byte value ) {
        _constructAll(ndConstructor, I8.class );
        ( (byte[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI64( NDConstructor ndConstructor, long value ) {
        _constructAll(ndConstructor, I64.class );
        ( (long[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllBool( NDConstructor ndConstructor, boolean value ) {
        _constructAll(ndConstructor, Boolean.class );
        ( (boolean[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllChar( NDConstructor ndConstructor, char value ) {
        _constructAll(ndConstructor, Character.class );
        ( (char[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAll( NDConstructor ndConstructor, Object value ) {
        _constructAll(ndConstructor, value.getClass() );
        ( (Object[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAll( NDConstructor ndConstructor, Class<?> typeClass ) {
        int size = ndConstructor.getSize();
        _API.setType( DataType.of( typeClass ) );
        configureFromNewShape(ndConstructor, size > 1, true );
    }

    private void _constructForDoubles( NDConstructor ndConstructor, double[] value )
    {
        int size = ndConstructor.getSize();
        _API.setType( DataType.of( F64.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (double[]) _API.getData())[ i ]  = value[ i % value.length ];
        }
        else _API.setData( value );
        configureFromNewShape(ndConstructor, false, false );
    }

    private void _constructForFloats( NDConstructor ndConstructor, float[] value )
    {
        int size = ndConstructor.getSize();
        _API.setType( DataType.of( F32.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (float[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape(ndConstructor, false, false );
    }

    private void _constructForInts( NDConstructor ndConstructor, int[] value )
    {
        int size = ndConstructor.getSize();
        _API.setType( DataType.of( I32.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (int[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape(ndConstructor, false, false );
    }

    private void _constructForShorts( NDConstructor ndConstructor, short[] value )
    {
        int size = ndConstructor.getSize();
        _API.setType( DataType.of( I16.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (short[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape(ndConstructor, false, false );
    }

    private void _constructForBooleans( NDConstructor ndConstructor, boolean[] value )
    {
        int size = ndConstructor.getSize();
        _API.setType( DataType.of( Boolean.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (boolean[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape(ndConstructor, false, false );
    }

    private void _constructForBytes( NDConstructor ndConstructor, byte[] value )
    {
        int size = ndConstructor.getSize();
        _API.setType( DataType.of( I8.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (byte[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape(ndConstructor, false, false );
    }

    private void _constructForLongs( NDConstructor ndConstructor, long[] value )
    {
        int size = ndConstructor.getSize();
        _API.setType( DataType.of( I64.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (long[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape(ndConstructor, false, false );
    }

    public <V> void constructSeeded( Class<V> valueType, NDConstructor ndConstructor, Object seed ) {
        _API.setType( DataType.of(valueType) );
        int size = ndConstructor.getSize();

        if ( valueType == Double.class )
            _API.setData( DataConverter.Utility.seededDoubleArray( new double[size], seed.toString() ) );
        else if ( valueType == Float.class )
            _API.setData( DataConverter.Utility.seededFloatArray( new float[size], seed.toString() ) );
        else if ( valueType == Integer.class )
            _API.setData( DataConverter.Utility.seededIntArray( new int[size], seed.toString() ) );
        else if ( valueType == Short.class )
            _API.setData( DataConverter.Utility.seededShortArray( new short[size], seed.toString() ) );
        else if ( valueType == Byte.class )
            _API.setData( DataConverter.Utility.seededByteArray( new byte[size], seed.toString() ) );
        else if ( valueType == Long.class )
            _API.setData( DataConverter.Utility.seededLongArray( new long[size], seed.toString() ) );
        else if ( valueType == Boolean.class )
            _API.setData( DataConverter.Utility.seededBooleanArray( new boolean[size], seed.toString() ) );
        else if ( valueType == Character.class )
            _API.setData( DataConverter.Utility.seededCharacterArray( new char[size], seed.toString() ) );
        else
            throw new IllegalArgumentException("Seeding not supported for value type '"+valueType.getSimpleName()+"'!");

        configureFromNewShape(ndConstructor, false, false  );
    }

    /**
     *  If possible, turns the provided {@code Object} array into a memory compact array of primitive types.
     *
     * @param dataType The {@link DataType} of the elements in the provided array.
     * @param values The array of values which ought to be optimized into a flat array of primitives.
     * @param size The size of the optimized array of primitives.
     * @return An optimized flat array of primitives.
     */
    private static Object _optimizeObjectArray( DataType<?> dataType, Object[] values, int size ) {
        Object data = values;
        IntStream indices = IntStream.iterate( 0, i -> i + 1 ).limit(size);
        if ( size > 1_000 ) indices = indices.parallel();
        indices = indices.map( i -> i % values.length );
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
        } else if ( dataType == DataType.of(Boolean.class) ) {
            boolean[] booleans = new boolean[size];
            for( int i = 0; i < size; i++ ) booleans[ i ] = (Boolean) values[ i % values.length ];
            data = booleans;
        } else if ( dataType == DataType.of(Character.class) ) {
            char[] chars = new char[size];
            for( int i = 0; i < size; i++ ) chars[ i ] = (Character) values[ i % values.length ];
            data = chars;
        }
        return data;
    }


}
